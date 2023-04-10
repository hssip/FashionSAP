import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from sklearn.metrics import f1_score


from models.model_fashion_catereg import FashionSAP
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
    metric_logger.add_meter('ce_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    lossfunc = nn.CrossEntropyLoss()

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = [i.to(device, non_blocking=True) for i in batch]
        (image, text_input_ids, text_attention_mask, outpos, label) = batch              
        cate_logit = model(image, text_input_ids, text_attention_mask, outpos)
        label = label.squeeze()
        loss = lossfunc(input=cate_logit, target=label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(ce_loss=loss.item())
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
    logits = []
    labels = []
    for i, batch in enumerate(metric_logger.log_every(data_loader, 50, header)):
        batch = [i.to(device) for i in batch]
        img, input_ids, attention_mask, outpos, label = batch
        #image, text_input_ids, text_attention_mask, outpos
        cate_logit = model(img, input_ids, attention_mask, outpos)
        label = label.squeeze()
        logits.append(cate_logit)
        labels.append(label)
   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    return logits.cpu().numpy(), labels.cpu().numpy()

@torch.no_grad()
def acc_eval(logits, labels, class_num=48):
    acc = 0
    micro_f = 0
    macro_f = 0
    pred = np.argmax(logits, axis=-1)
    acc = np.equal(pred, labels).astype(np.float64).mean()
    micro_f = f1_score(y_true=labels, y_pred=pred, average='micro')
    macro_f = f1_score(y_true=labels, y_pred=pred, average='macro')
    eval_result =  {'acc':acc,
                    'macro_f':macro_f,
                    'micro_f':micro_f}
    
    return eval_result

def main(args, config):
    if args.cate_kind == 'cate':
        args.class_num = 48
    else:
        args.class_num = 121

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
    train_dataset, test_dataset = create_dataset('catereg', config, args,tokenizer)  


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader = create_loader([train_dataset, test_dataset],samplers,
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
        if 'queue_size' in config and config['queue_size'] < 65535:
            state_dict['image_queue'] = state_dict['image_queue'][:,:config['queue_size']]
            state_dict['text_queue'] = state_dict['text_queue'][:,:config['queue_size']]
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        
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
        model_state_dict = checkpoint['model']

        msg = model.load_state_dict(model_state_dict,strict=True)  
        print('load checkpoint from %s'%args.checkpoint)
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
        logits, labels = evaluation(model_without_ddp, val_loader, tokenizer, device, config, args)

    
        if utils.is_main_process():   
            val_result = acc_eval(logits, labels, args)
            print(val_result)

            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},                 
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},                 
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
                if val_result['acc']>best:
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
                    best = val_result['acc']    
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
    parser.add_argument('--config', default='./configs/fashion_catereg.yaml')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--pre_point', default='checkpoint_best.pth') 
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--sub_dataset', default=False, type=bool)

    parser.add_argument('--max_word_num', default=75, type=int)
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--cate_kind', default='cate', type=str)
    parser.add_argument('--prompt', default=False, type=bool)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
