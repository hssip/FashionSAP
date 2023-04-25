
import os
from torch.utils.data import Dataset
import torch

from PIL import Image
import h5py
import ujson as json
import numpy as np
from .utils import pre_caption

# 48 cates
CATE_LIST = [
'tote bags', 'blankets', 'jackets & coats', 'boat shoes & moccasins', 'lace ups', 'loafers', 'shoulder bags', 'briefcases', 
'ties', 'pants', 'dresses', 'pocket squares & tie bars', 'travel bags', 'hats', 'shorts', 'gloves', 'underwear & loungewear', 
'scarves', 'suits & blazers', 'keychains', 'fine jewelry', 'socks', 'messenger bags', 'duffle bags', 'boots', 'duffle & top handle bags', 
'sweaters', 'eyewear', 'shirts', 'flats', 'sneakers', 'messenger bags & satchels', 'backpacks', 'clutches & pouches', 'sandals', 'jumpsuits', 
'jewelry', 'tops', 'skirts', 'bag accessories', 'lingerie', 'jeans', 'espadrilles', 'belts & suspenders', 'swimwear', 'heels', 'pouches & document holders', 'monkstraps']

#121 subcates
SUBCATE_LIST = [
'structured hats', 'boy shorts', 'monkstraps', 'lace ups & oxfords', 'messenger bags', 'fur & shearling', 'peacoats', 
'beach hats', 'heeled sandals', 'trousers', 'crewnecks', 'denim jackets', 'heels', 'mid length skirts', 'necklaces', 
'one-piece', 'waistcoats', 'boat shoes & moccasins', 'mid length dresses', 'vests', 'loafers', 'biker & combat boots', 
'beanies', 'short dresses', 'jackets', 'messenger bags & satchels', 'long dresses', 'backpacks', 'leather pants', 'tank tops & camisoles', 
'sandals', 'turtlenecks', 'jumpsuits', 'pins', 'shawlnecks', 'henleys', 'bodysuits', 'fedoras & panama hats', 'cardigans', 'brooches', 'shorts', 
'bow ties', 'blazers', 'tank tops', 'aviator', 'blankets', 'sleepwear', 'thongs', 'caps', 'bombers', 'clutches', 'flat sandals', 'low top sneakers', 
'ankle boots', 'sweatshirts', 'v-necks', 'shirts', 'ballerina flats', 'earrings', 'rings', 'hoodies & zipups', 'sunglasses', 'sweatpants', 
'high top sneakers', 'trench coats', 'cargo pants', 'swimsuits', 'duffle bags', 'bras', 'coats', 'pouches', 'espadrilles', 
'caps & flat caps', 'mid-calf boots', 'glasses', 'silks & cashmeres', 'briefcases', 'short skirts', 'long skirts', 'tall boots', 
'jeans', 'neck ties', 'keychains', 'slippers & loafers', 'wedge sneakers', 'leggings', 'tanks', 'flip flops', 'cover ups', 
'duffle & top handle bags', 't-shirts', 'socks', 'leather jackets', 'travel bags', 'gloves', 'tote bags', 'knits', 'blouses', 
'lounge pants', 'bracelets', 'zip up & buckled boots', 'polos', 'robes', 'headbands & hair accessories', 'belts & suspenders', 
'wingtip boots', 'desert boots', 'boxers', 'shoulder bags', 'bikinis', 'down', 'suits', 'tuxedos', 'lace-up boots', 
'chelsea boots', 'pouches & document holders', 'pocket squares & tie bars', 'lace ups', 'scarves', 'briefs', 'bag accessories'
]

class fashiongen_dataset_catereg(Dataset):
    def __init__(self, args, transform, tokenizer, split):   
        self.data_root = args.data_root
        # self.sub_dataset = args.sub_dataset
        self.split = 'validation' if split in ('val', 'test') else split
        self.max_words = args.max_word_num
        self.prompt = args.prompt
        self.cate = 'category' if args.cate_kind == 'cate' else 'sub category'
        self.cate_list = CATE_LIST if args.cate_kind == 'cate' else SUBCATE_LIST
        assert args.class_num == len(self.cate_list)

        self.transform = transform
        self.info_data, self.raw_data= self.__load_datafile()

        #list of dict the dict is text info of instance
        #h5py data of raw data

        self.tokenizer = tokenizer
        self.label_map = {k:i for i,k in enumerate(self.cate_list)}


    def __load_datafile(self):
        datafilename = os.path.join(self.data_root, self.split, 'info_data.json')
        infodata = json.load(open(datafilename, mode='r', encoding='utf-8'))
        raw_datafilename = os.path.join(self.data_root, 'fashiongen_256_256_{}.h5'.format(self.split))
        raw_data = h5py.File(raw_datafilename)
        assert len(infodata) == len(raw_data['index_2'])
        return infodata, raw_data

    def __get_image(self, index):
        img = Image.fromarray(self.raw_data['input_image'][index])
        img = self.transform(img)
        return img

    def __get_text(self, index):
        item_info = self.info_data[index]
        ###### process description
        if not self.prompt:
            description = 'the image description is ' + item_info['input_description']
            description_tokens = self.tokenizer.tokenize(pre_caption(description))
            description_tokens = ['[CLS]'] + description_tokens
            tokens_length = len(description_tokens)
            if tokens_length > self.max_words:
                description_tokens = description_tokens[:self.max_words]
                tokens_length = self.max_words
            outpos = [0]
        else:
            description = 'the image description is ' + item_info['input_description']
            description_tokens = ['[CLS]'] + self.tokenizer.tokenize(pre_caption(description))
            tokens_length = len(description_tokens)
            if (tokens_length + 5) > self.max_words:
                description_tokens = description_tokens[:(self.max_words - 5)]
            description_tokens.extend(self.tokenizer.tokenize('the image ' + self.cate + ' is ') + ['[MASK]'])
            tokens_length = len(description_tokens)
            assert tokens_length < self.max_words or tokens_length == self.max_words
            outpos = [len(tokens_length) - 1]


        pad_length = self.max_words - tokens_length
        input_ids = self.tokenizer.convert_tokens_to_ids(description_tokens)
        attention_mask = [1]* len(input_ids) + [0] * pad_length
        input_ids = input_ids + [0] * pad_length

        ####change list to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        outpos = torch.tensor(outpos, dtype=torch.long)
        return input_ids, attention_mask, outpos
        
    def __len__(self):

        return len(self.info_data)
    
    def __getitem__(self, index):
        cap_index = index
        input_ids, attention_mask, outpos = self.__get_text(cap_index)
        img = self.__get_image(cap_index)
        label = self.__get_label(cap_index)
        return img, input_ids, attention_mask, outpos, label
    
    def __get_label(self, index):
        cate_label = 'input_category' if self.cate =='cate' else 'input_subcategory'
        cate = self.info_data[index][cate_label].strip().replace('\t',' ').lower()
        label = torch.tensor([self.label_map.get(cate, 0)], dtype=torch.long)
        return label
