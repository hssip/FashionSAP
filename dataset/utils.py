
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