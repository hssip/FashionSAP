
import re
from nltk.corpus import wordnet as wn
import random

def process_description(sent, tagger=None, tokenizer=None):

    sent_tokens = tokenizer.tokenize(pre_caption(sent))
    tokenizer_vocab = list(tokenizer.vocab.keys())

    tokens_length = len(sent_tokens)
    mask_token = []
    mask_sign = [0]*tokens_length
    replace_token = []
    replace_sign = [0]*tokens_length
    
    pos_index = list(range(tokens_length))
    random.shuffle(pos_index)
    tokenizer_vocab = ['[MASK]'] if tokenizer_vocab is None else tokenizer_vocab

    #### mask sentence
    mask_length = int(tokens_length * 0.15 * 0.8)
    for i in range(mask_length):
        mask_token.append(sent_tokens[pos_index[i]])
        mask_sign[pos_index[i]] = 1
        sent_tokens[pos_index[i]] = '[MASK]'

    replace_length = int(tokens_length * 0.15 * 0.1)
    for i in range(mask_length, mask_length+replace_length):
        mask_token.append(sent_tokens[pos_index[i]])
        mask_sign[pos_index[i]] = 1

        replace_token.append(sent_tokens[pos_index[i]])
        replace_sign[pos_index[i]] = 1
        if random.random() < 0.5:
            sent_tokens[pos_index[i]] = random.choice(tokenizer_vocab)
        else:
            sent_tokens[pos_index[i]] = search_antonym(sent_tokens[pos_index[i]])
    added_tokens = tokenizer.tokenize('the image description is')
    sent_tokens = added_tokens + sent_tokens
    replace_sign = [0]* (len(added_tokens)) + replace_sign
    mask_sign = [0] * (len(added_tokens)) + mask_sign


    return sent_tokens, mask_token, mask_sign ,replace_token, replace_sign


    

def process_attribute(attribute_dict, tokenizer) -> list:
    outstrs = []
    attribute_names = list(attribute_dict.keys())
    random.shuffle(attribute_names)
    vocabs = list(tokenizer.vocab.keys())
    if len(attribute_dict) == 1:
        yes_flag = 1 if random.random() < 0.5 else 0
        replace_flag = 1 if random.random() < 0.5 else 0
        val = attribute_dict[attribute_names[0]]
        if replace_flag == 0:
            outstr = 'the image attribute is ' + val
            out_tokens = tokenizer.tokenize(outstr)
            mask_token = []
            
            if yes_flag:
                out_tokens = 'is the image attribute ' + val + ' ? '
                out_tokens = tokenizer.tokenize(out_tokens)
                out_tokens += ['[MASK]']
                mask_token = ['yes']
            else:
                neg_val = ' '.join([random.choice(vocabs) for _ in val.split(' ')])
                out_tokens = 'is the image attribute ' + neg_val + ' ? '
                out_tokens = tokenizer.tokenize(out_tokens)
                out_tokens += ['[MASK]']
                mask_token = ['no']
        mask_sign = [0 if t != '[MASK]' else 1 for t in out_tokens]
        replace_sign = [0] * len(out_tokens)
        replace_token = []
        
        return out_tokens, mask_token, mask_sign, replace_token, replace_sign
    
    ##replace name this part is not used in experiment
    # name = attribute_names[0]
    # val = attribute_dict[name]
    # name = search_synonym(name.replace('input_',''))

    #replace attribute value
    name = attribute_names[0]
    val = attribute_dict[name]
    val_tokens = val.split(' ')
    val = ''
    for vt in val_tokens:
        val += search_synonym(vt) + ' '
    outstrs.append((name.replace('input_',''),val))

    for name in attribute_names[1:]:
        outstrs.append((name.replace('input_',''), attribute_dict[name]))
    # print(outstrs)
    ##### process
    for i, v in enumerate(outstrs):
        if v[0] == 'subcategory':
            outstrs[i] = ('sub category',v[1])

    out_tokens = []
    for attr in outstrs:
        name, val = attr
        out_tokens.append(tokenizer.tokenize(name.lower()))
        out_tokens.append(tokenizer.tokenize(val.replace('&','and').lower()))
    prompt_tokens = tokenizer.tokenize('the image ')
    #format out_tokens now is a list of squence[name1, val1, name2, val2,...]
    
    ##### prompt name or attribute
    mask_token = []
    prompt_val_flag = 0 if random.random() < 0.5 else 1
    # if prompt_val_flag == 0:
    mask_token = out_tokens[4+prompt_val_flag]    # 4 because skip first and second (name-val) tuple
    out_tokens[4+prompt_val_flag] = ['[MASK]'] * len(out_tokens[4+prompt_val_flag])
            
        
    for i in range(0, len(out_tokens),2):
        out_tokens[i] = prompt_tokens + out_tokens[i] + ['is']
    
    
    out_tokens = [i for k in out_tokens for i in k]
    mask_sign = [0]* len(out_tokens)
    for i,token in enumerate(out_tokens):
        if token != '[MASK]':
            continue
        else:
            mask_sign[i] = 1
    replace_token = []
    replace_sign = [0] * len(out_tokens)


    return out_tokens, mask_token, mask_sign, replace_token, replace_sign
        

def search_synonym(word, label=None):
    '''
    if finded return syn else return word
    '''
    assert label in ('n','a','v',None)
    syns = wn.synsets(word)
    syns_set = set()
    for syn in syns:
        if label is not None and \
            syn.name().split('.')[1] != label:
            continue
        syns_set | set(syn.lemma_names())
    syns_set.discard(word)
    if syns_set:
        word = random.choice(list(syns_set))

    return word




def search_antonym(word, label=None):
    anto = []
    for syn in wn.synsets(word):
        for lm in syn.lemmas():
            if lm.antonyms():
                anto.append(lm.antonyms()[0].name())
    return random.choice(anto) if anto else word
        



def pre_caption(caption,max_words=None):
    caption = re.sub(
        # r"([,.'!?\"()*#:;~])",
        r"([,.'!\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person').replace('<br>',' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    if max_words is not None:
        caption_words = caption.split(' ')
        if len(caption_words)>max_words:
            caption = ' '.join(caption_words[:max_words])
            
    return caption

def pharse_fashiongen_season(input_season):
    '''
    pharse season abbreviation to origin string tuble for fashiongen
    '''
    assert len(input_season) == 6
    season = input_season[:2]
    year = input_season[2:]
    assert season in ('SS','FW')
    if season == 'SS':
        season = 'spring summer'
    elif season == 'FW':
        season = 'fall winter'
    return season + ' ' + year


# from vqaTools.vqaEval import VQAEval
# from refTools.evaluation.refEvaluation import RefEvaluation

# import json
# import os
# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.nn.functional as F

# import utils
# from tqdm import tqdm


# def vqa_eval(vqa, result_file, test_ques_path):
#     vqaRes = vqa.loadRes(result_file, test_ques_path)
#     # create vqaEval object by taking vqa and vqaRes
#     vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
#     # evaluate results
#     vqaEval.evaluate()   

#     # print accuracies
#     print("\n")
#     print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
#     print("Per Answer Type Accuracy is the following:")
#     for ansType in vqaEval.accuracy['perAnswerType']:
#         print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
#     print("\n")    
    
#     return vqaEval


# def collect_result(result, result_dir, filename, is_json=True, is_list=True):
#     if is_json:
#         result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
#         final_result_file = os.path.join(result_dir, '%s.json'%filename)
#         json.dump(result,open(result_file,'w'))
#     else:
#         result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
#         final_result_file = os.path.join(result_dir, '%s.pth'%filename)
#         torch.save(result,result_file)     
        
#     dist.barrier()
    
#     result = None
#     if utils.is_main_process():   
#         # combine results from all processes
#         if is_list:
#             result = []
#         else:
#             result = {}
#         for rank in range(utils.get_world_size()):
#             if is_json:
#                 result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
#                 res = json.load(open(result_file,'r'))
#             else:
#                 result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
#                 res = torch.load(result_file)            
#             if is_list:
#                 result += res
#             else:
#                 result.update(res) 
      
#     return result    

    
# def save_result(result, result_dir, filename, is_json=True, is_list=True):
#     if is_json:
#         result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
#         final_result_file = os.path.join(result_dir, '%s.json'%filename)
#         json.dump(result,open(result_file,'w'))
#     else:
#         result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
#         final_result_file = os.path.join(result_dir, '%s.pth'%filename)
#         torch.save(result,result_file)     
        
#     dist.barrier()

#     if utils.is_main_process():   
#         # combine results from all processes
#         if is_list:
#             result = []
#         else:
#             result = {}
#         for rank in range(utils.get_world_size()):
#             if is_json:
#                 result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
#                 res = json.load(open(result_file,'r'))
#             else:
#                 result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
#                 res = torch.load(result_file)            
#             if is_list:
#                 result += res
#             else:
#                 result.update(res)
#         if is_json:                  
#             json.dump(result,open(final_result_file,'w'))   
#         else:            
#             torch.save(result,final_result_file)     
        
#         print('result file saved to %s'%final_result_file)
#     dist.barrier()        
#     return final_result_file


# def grounding_eval(results,dets,cocos,refer,alpha,mask_size=24):
    
#     correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
#     correct_A, correct_B, correct_val = 0, 0, 0 
#     num_A,num_B,num_val = 0,0,0
    
#     for res in tqdm(results):

#         ref_id = res['ref_id']
#         ref = refer.Refs[ref_id]
#         ref_box = refer.refToAnn[ref_id]['bbox']
#         image = refer.Imgs[ref['image_id']]

#         mask = res['pred'].cuda().view(1,1,mask_size,mask_size)    
#         mask = F.interpolate(mask,size = (image['height'],image['width']), mode='bicubic').squeeze()
        
#         # rank detection boxes
#         max_score = 0
#         for det in dets[str(ref['image_id'])]:
#             score = mask[int(det[1]):int(det[1]+det[3]),int(det[0]):int(det[0]+det[2])]
#             area = det[2]*det[3]
#             score = score.sum() / area**alpha
#             if score>max_score:
#                 pred_box = det[:4]
#                 max_score = score    

#         IoU_det = computeIoU(ref_box, pred_box)
        
#         if ref['split']=='testA':
#             num_A += 1    
#             if IoU_det >= 0.5:   
#                 correct_A_d += 1            
#         elif ref['split']=='testB':
#             num_B += 1    
#             if IoU_det >= 0.5:   
#                 correct_B_d += 1    
#         elif ref['split']=='val':
#             num_val += 1    
#             if IoU_det >= 0.5:   
#                 correct_val_d += 1    
                
#     eval_result = {'val_d':correct_val_d/num_val,'testA_d':correct_A_d/num_A,'testB_d':correct_B_d/num_B}        
    
#     for metric, acc in eval_result.items():
#         print(f'{metric}: {acc:.3f}')
        
#     return eval_result    

# # IoU function
# def computeIoU(box1, box2):
#     # each box is of [x1, y1, w, h]
#     inter_x1 = max(box1[0], box2[0])
#     inter_y1 = max(box1[1], box2[1])
#     inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
#     inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

#     if inter_x1 < inter_x2 and inter_y1 < inter_y2:
#         inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
#     else:
#         inter = 0
#     union = box1[2]*box1[3] + box2[2]*box2[3] - inter
#     return float(inter)/union

        
        