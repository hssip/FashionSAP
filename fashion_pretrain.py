import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.distributed as dist




from models.model_fashion_pretrain import FashionSAP
from models.vit import interpolate_pos_embed
from transformers.models.bert.tokenization_bert import BertTokenizer


import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('sis', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('ml', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('rl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = [i.to(device, non_blocking=True) for i in batch]
        (image, text_input_ids, text_attention_mask, mask_labels, replace_labels, idx) = batch
            
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        # loss_ita, loss_itm = model(image, text_input,alpha=alpha, idx=idx)        
        idx = None          
        loss_ita, loss_itm, symbol_simloss, mask_loss, replace_loss = model(image, text_input_ids, text_attention_mask,alpha=alpha, 
                                mask_labels=mask_labels, replace_labels=replace_labels)
                          
        loss = loss_ita + loss_itm + 0. * symbol_simloss + mask_loss + 0. * replace_loss
        # loss = loss_itm + symbol_simloss + mask_loss + replace_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(itm=loss_itm.item())
        metric_logger.update(ita=loss_ita.item())
        metric_logger.update(sis=symbol_simloss.item())
        metric_logger.update(ml=mask_loss.item())
        metric_logger.update(rl=replace_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    
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
    #### Dataset #### 
    print("Creating dataset")
    train_dataset, test_dataset = create_dataset('pretrain', config, args,tokenizer)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None, None]
    
    train_loader, test_loader = create_loader([train_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4],
                                                          is_trains=[True, False], 
                                                          collate_fns=[None,None])   
       

    #### Model #### 
    print("Creating model")
    model = FashionSAP(config=config, args=args)


    if args.pre_point:
        checkpoint = torch.load(args.pre_point, map_location='cpu')    
        state_dict = checkpoint['model']
        # if config['queue_size'] < 65536:
        #     state_dict['image_queue'] = state_dict['image_queue'][:,:config['queue_size']]
        #     state_dict['text_queue'] = state_dict['text_queue'][:,:config['queue_size']]
        
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
    elif args.bert_point and args.vit_point:
        t_checkpoint = torch.load(args.bert_point, map_location='cpu')
        t_checkpoint = utils.text_state_compatibility(t_checkpoint)
        msg = model.text_encoder.load_state_dict(t_checkpoint, strict=False)
        print('load checkpoint from %s'%args.bert_point)
        print(msg)
        
        v_checkpoint = torch.load(args.vit_point, map_location='cpu')['model']
        pos_embed_reshaped = interpolate_pos_embed(v_checkpoint['pos_embed'],model.visual_encoder)         
        v_checkpoint['pos_embed'] = pos_embed_reshaped
        for k in list(v_checkpoint.keys()):
            if k.startswith('head.'):
                del v_checkpoint[k]
        msg = model.visual_encoder.load_state_dict(v_checkpoint, strict=True)
        print('load checkpoint from %s'%args.vit_point)
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
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    print(args)
    print(config)
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)
            print(train_stats)   
                    
            if epoch > int(max_epoch/2):
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                outpath = os.path.join(args.output_dir,'epcoh'+str(epoch))
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                torch.save(save_obj, os.path.join(outpath, 'checkpoint_best.pth'))  
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
              

            
if __name__ == '__main__':
    # print('without loss sis')
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/fashion_pretrain.yaml')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--pre_point', default='ALBEF.pth')
    # parser.add_argument('--bert_point', default='bert-base-uncased-pytorch_model.bin')
    # parser.add_argument('--vit_point', default='deit_base_patch16_224-b5f2ef4d.pth')
    
    
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    # parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    # parser.add_argument('--sub_dataset', default=False, type=bool)

    parser.add_argument('--max_word_num', default=180, type=int)
    parser.add_argument('--data_root', default='/mnt/MDisk/hyp/data/fashiongen', type=str)
    parser.add_argument('--catemap_filename', default='categorys_to_sign.txt', type=str)
    parser.add_argument('--product_list_filename', default='productid_list.json', type=str)
    parser.add_argument('--replace_kind_num', default=2, type=int)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
