import argparse
import os
from statistics import mode
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from models.model_fashion_tgir import FashionSAP
from models.vit import interpolate_pos_embed
from transformers.models.bert.tokenization_bert import BertTokenizer



import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    # for i,(image, text_input_ids, text_attention_mask, text_token_types, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = [i.to(device, non_blocking=True) for i in batch]
        (text_input_ids, text_attention_mask, reference_img, target_img, ref_id, tar_id, cap_index) = batch
            
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        # ref_image, tar_image, text_input_ids, text_attention_mask, alpha                 
        loss_ita= model(text_input_ids, text_attention_mask, reference_img, target_img, alpha=alpha, ref_id=ref_id, tar_id=tar_id, cap_index=cap_index)                  
        loss = loss_ita
        # loss = loss_itm + symbol_simloss + mask_loss + replace_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(ita=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, args):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    fusion_feats = []
    text_len = len(data_loader.dataset.texts)
    img_len = len(data_loader.dataset.imgs)
    img_feats = torch.full((img_len, config['embed_dim']), -100.0).to(device)
    for i, batch in enumerate(metric_logger.log_every(data_loader, 50, header)):
        # print(batch)
        batch = [k.to(device) for k in batch]
        text_input_ids, text_attention_mask, reference_img, target_img, label = batch
        text_output = model.text_encoder(text_input_ids, text_attention_mask, return_dict = True, mode='text')
        ref_img_output = model.visual_encoder(reference_img)
        tar_img_output = model.visual_encoder(target_img)
        cross_hidden = ref_img_output
        cross_att = torch.ones(cross_hidden.size()[:-1], dtype=torch.long).to(device)
        fusion_output = model.text_encoder(
                encoder_embeds = text_output.last_hidden_state, 
                attention_mask = text_attention_mask,
                encoder_hidden_states = ref_img_output,
                encoder_attention_mask = cross_att,                             
                return_dict = True,
                mode = 'fusion'
        )
        # print(label.size())
        ref_pos = torch.flatten(label[:,1])
        tar_pos = torch.flatten(label[:,2])
        fusion_feat = F.normalize(model.combine_text_proj(model.text_proj(fusion_output.last_hidden_state[:,0,:])), dim=-1)
        ref_img_feat = F.normalize(model.combine_vision_proj(model.vision_proj(ref_img_output[:,0,:])),dim=-1)
        tar_img_feat = F.normalize(model.combine_vision_proj(model.vision_proj(tar_img_output[:,0,:])),dim=-1)
        
        fusion_feats.append(fusion_feat)
        img_feats[ref_pos] = ref_img_feat
        img_feats[tar_pos] = tar_img_feat
    
    fusion_feats = torch.cat(fusion_feats, dim=0)
    img_feats = img_feats
    
    sim_f2i = fusion_feats @ img_feats.t()
    sim_i2f = None
    
    
 


        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    # match_i2t = match_i2t.cpu().numpy() if match_i2t is not None else None
    # match_t2i = match_t2i.cpu().numpy() if match_t2i is not None else None
    sim_i2f = sim_i2f.cpu().numpy() if sim_i2f is not None else None
    sim_f2i = sim_f2i.cpu().numpy() if sim_f2i is not None else None


    # return sim_t2i.T.cpu().numpy(), sim_t2i.cpu().numpy(), match_i2t, match_t2i
    return sim_f2i, sim_i2f


            
@torch.no_grad()
def itm_eval(sim_f2i, sim_i2f, labels):
    # print(len(labels))
    # print(sim_f2i.shape)
    assert len(labels) == sim_f2i.shape[0]
    labels = [i[2] for i in labels]
    ranks = np.zeros(sim_f2i.shape[0])
    for index,score in enumerate(sim_f2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == labels[index])[0][0]
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    ir50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    
    
    if sim_i2f is not None:
        pass

    # tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    # r_mean = (tr_mean + ir_mean) / 2
    # r_mean = (tr1 + ir1) / 2

    eval_result =  {'ir1': ir1,
                    'ir5': ir5,
                    'ir10': ir10,
                    'ir50': ir50,
                    'img_r_mean': ir_mean,}
    return eval_result




def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    #### process tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_config'])
    inner_slots = [
            '[tops_sign]',
            '[pants_sign]',
            '[skirts_sign]',
            '[dresses_sign]',
            '[coats_sign]',
            '[shoes_sign]',
            '[bags_sign]',
            '[accessories_sign]',
            '[others_sign]',
    ]
    tokenizer.add_tokens(inner_slots, special_tokens=True)
    #### tokenizer done ####
    train_dataset, dress_test_dataset, toptee_test_dataset, shirt_test_dataset, all_test_dataset, origin_test_dataset = create_dataset('tgir', config, args,tokenizer)  
    # a = test_dataset.get_test_labels()
    # tiny_i2t_img, tiny_i2t_txt = test_dataset.get_i2t_test()
    # tiny_t2i_txt, tiny_t2i_img = test_dataset.get_i2t_test()

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]*4
    else:
        samplers = [None, None, None, None, None]
    
    train_loader, dress_loader, toptee_loader, shirt_loader, all_test_loader = create_loader([train_dataset, dress_test_dataset, toptee_test_dataset, shirt_test_dataset, all_test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*4,
                                                          num_workers=[4,4,4,4,4],
                                                          is_trains=[True, False, False, False, False],
                                                          collate_fns=[None]*len(samplers))   
       

    #### Model #### 
    print("Creating model")
    model = FashionSAP(config=config, args=args)



    if args.pre_point:
        checkpoint = torch.load(args.pre_point, map_location='cpu')    
        state_dict = checkpoint['model']
        new_leng = config['queue_size']
        pre_leng = state_dict['image_queue'].size()[1]
        if new_leng < pre_leng:
            state_dict['image_queue'] = state_dict['image_queue'][:,:new_leng]
            state_dict['fusion_queue'] = state_dict['fusion_queue'][:,:new_leng] 
        elif new_leng > pre_leng:
            state_dict['image_queue'] = torch.cat([state_dict['image_queue'],state_dict['image_queue'][:,:new_leng-pre_leng]], dim=1)
            state_dict['fusion_queue'] = torch.cat([state_dict['fusion_queue'],state_dict['fusion_queue'][:,:new_leng-pre_leng]], dim=1)   
        state_dict.pop('queue_ptr')
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.pre_point)
        print(msg)
    elif args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        new_leng = config['queue_size']
        pre_leng = state_dict['image_queue'].size()[1]
        if new_leng < pre_leng:
            state_dict['image_queue'] = state_dict['image_queue'][:,:new_leng]
            state_dict['fusion_queue'] = state_dict['fusion_queue'][:,:new_leng] 
        elif new_leng > pre_leng:
            state_dict['image_queue'] = torch.cat([state_dict['image_queue'],state_dict['image_queue'][:,:new_leng-pre_leng]], dim=1)
            state_dict['fusion_queue'] = torch.cat([state_dict['fusion_queue'],state_dict['fusion_queue'][:,:new_leng-pre_leng]], dim=1)   
        state_dict.pop('queue_ptr')
        # state_dict.pop('idx_queue')
        # state_dict['fusion_queue'] = state_dict.pop('text_queue')
        # if config['queue_size'] < 65536:
        #     state_dict['image_queue'] = state_dict['image_queue'][:,:config['queue_size']]
        #     state_dict['fusion_queue'] = state_dict['fusion_queue'][:,:config['queue_size']]
        # state_dict.pop('queue_ptr')
        # pop_pars = ['combine_text_proj','combine_vision_proj']
        # pop_keys = []
        # for key in state_dict.keys():
        #     for pk in pop_pars:
        #         if pk in key:
        #             pop_keys.append(key)
        # for pk in pop_keys:
        #     state_dict.pop(pk)

        msg = model.load_state_dict(state_dict,strict=False)  
        # model.copy_params()
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)
    elif args.newcheckpoint:
        checkpoint = torch.load(args.newcheckpoint, map_location='cpu')
        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
        state_dict = checkpoint['model']

        msg = model.load_state_dict(state_dict,strict=True)  
        # model.copy_params()
        print('load checkpoint from %s'%args.newcheckpoint)
        print(msg)

        
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.device != 'cpu' and args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    # lr_scheduler.load_state_dict()
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    print(args)
    print(config)
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            
        # score_f2i, score_i2f= evaluation(model_without_ddp, test_loader, tokenizer, device, config, args)
        dress_score_f2i, score_i2f= evaluation(model_without_ddp, dress_loader, tokenizer, device, config, args)
        toptee_score_f2i, score_i2f= evaluation(model_without_ddp, toptee_loader, tokenizer, device, config, args)
        shirt_score_f2i, score_i2f= evaluation(model_without_ddp, shirt_loader, tokenizer, device, config, args)
        alltest_score_f2i, score_i2f= evaluation(model_without_ddp, all_test_loader, tokenizer, device, config, args)
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
    
        if utils.is_main_process():  
            # labels = test_dataset.get_labels()
            dress_labels = dress_test_dataset.get_labels()
            toptee_labels = toptee_test_dataset.get_labels()
            shirt_labels = shirt_test_dataset.get_labels()
            all_labels = all_test_dataset.get_labels()
            # img2text, text2img = test_dataset.get_test_labels()
            # tiny_i2t_img, tiny_i2t_txt = test_dataset.get_i2t_test()
            # tiny_t2i_txt, tiny_t2i_img = test_dataset.get_t2i_test()
            dress_val_result = itm_eval(dress_score_f2i, score_i2f, dress_labels)  
            toptee_val_result = itm_eval(toptee_score_f2i, score_i2f, toptee_labels)  
            shirt_val_result = itm_eval(shirt_score_f2i, score_i2f, shirt_labels)  
            all_test_val_result = itm_eval(alltest_score_f2i, score_i2f, all_labels)  
            print('******* dress val result ********')
            print(dress_val_result)
            print('******* toptee val result ********')
            print(toptee_val_result)
            print('******* shirt val result ********')
            print(shirt_val_result)
            print('******* all val result ********')
            print(all_test_val_result)
            # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)    
            # print(test_result)
            
            if args.evaluate:                
                log_stats = {**{f'dress_val_{k}': v for k, v in dress_val_result.items()},
                             **{f'toptee_val_{k}': v for k, v in toptee_val_result.items()},                  
                             **{f'shirt_val_{k}': v for k, v in shirt_val_result.items()},                  
                             **{f'all_val_{k}': v for k, v in shirt_val_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'dress_val_{k}': v for k, v in dress_val_result.items()},
                             **{f'toptee_val_{k}': v for k, v in toptee_val_result.items()},
                             **{f'shirt_val_{k}': v for k, v in shirt_val_result.items()},
                             **{f'all_val_{k}': v for k, v in shirt_val_result.items()},
                            #  **{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
                if dress_val_result['ir10'] +toptee_val_result['ir10'] +shirt_val_result['ir10'] >best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    # outpath = os.path.join(args.output_dir,'epcoh'+str(epoch))
                    outpath = args.output_dir
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)
                    torch.save(save_obj, os.path.join(outpath, 'checkpoint_best.pth'))  
                    best = dress_val_result['ir10'] +toptee_val_result['ir10'] +shirt_val_result['ir10']    
                    best_epoch = epoch
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/fashion_tgir.yaml')
    parser.add_argument('--output_dir', default='/mnt/MDisk/hyp/out_albef/fashion_tgir/nosis')
    # parser.add_argument('--pre_point', default='/mnt/MDisk/hyp/predata/albef/ALBEF.pth')
    parser.add_argument('--pre_point', default='')   
    # parser.add_argument('--checkpoint', default='')   
    # parser.add_argument('--checkpoint', default='/data/hyp/myfashion/pretrain_new/checkpoint_best.pth')   
    # parser.add_argument('--checkpoint', default='/mnt/MDisk/hyp/out_albef/fashiongen_retrieval/only_bef_raw/checkpoint_best.pth')   
    # parser.add_argument('--checkpoint', default='/mnt/MDisk/hyp/out_albef/fashiongen_pretrain_ablation/epcoh27/checkpoint_best.pth')   
    parser.add_argument('--checkpoint', default='/mnt/MDisk/hyp/out_albef/fashiongen_pretrain_ablation/nosis/epcoh26/checkpoint_best.pth')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate_match', action='store_true')
    parser.add_argument('--device', default='cuda')
    # parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--sub_dataset', default=False, type=bool)

    parser.add_argument('--max_word_num', default=79, type=int)
    parser.add_argument('--data_root', default='/mnt/MDisk/hyp/data/fashioniq', type=str)
    parser.add_argument('--train_class', default='all', type=str)

    
    


    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    # if arg.evaluate:
    #     args.output_dir = 
    
    main(args, config)
