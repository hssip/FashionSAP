
import os
from torch.utils.data import Dataset
import torch
from copy import deepcopy

from PIL import Image
import h5py
import ujson as json
import numpy as np
from .utils import pharse_fashiongen_season, pre_caption, process_description, process_attribute
import random

class fashiongen_dataset_pretrain(Dataset):
    def __init__(self, args, transform, tokenizer, split):   
        self.data_root = args.data_root
        self.split = 'validation' if split in ['val', 'test'] else split
        self.max_words = args.max_word_num

        self.catemap_filename = os.path.join(self.data_root,args.catemap_filename)
        self.product_list_filename = os.path.join(self.data_root, self.split, args.product_list_filename)
        self.__inner_attribute_names = [  
            'input_category',
            'input_composition',
            'input_gender',
            'input_name',
            'input_season',
            'input_subcategory']

        self.transform = transform
        self.info_data, self.raw_data= self.__load_datafile()
        #list of dict the dict is text info of instance
        #h5py data of raw data

        self.product_list, self.product_indexmap = self.__load_product_info()
        #list of int, the int is productid
        #dict of product index, key is productid, value is a list of product-info index
        self.cap_ids = self.__load_capids()
        self.catemap = self.__load_cate_map()   
        #dict map category val to sign token
        self.tokenizer = tokenizer
        self.vocabs = list(self.tokenizer.vocab.keys())


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
        description = item_info['input_description']
        description_tokens, des_mask_token, des_mask_sign, des_replace_token, des_replace_sign = \
            process_description(description, tagger=None, tokenizer=self.tokenizer)
        
        #process attribute
        attributes = {k:item_info[k] for k in self.__inner_attribute_names}
        attributes['input_season'] = pharse_fashiongen_season(attributes['input_season'])
        attribute_tokens, attr_mask_token, attr_mask_sign, attr_replace_token, attr_replace_sign= process_attribute(attributes, self.tokenizer)
        
        #process all
        sign_token = self.catemap[attributes['input_category'].lower()]
        all_tokens = ['[CLS]',sign_token] + description_tokens + ['[SEP]'] + attribute_tokens
        mask_sign = [0,0] + des_mask_sign + [0] + attr_mask_sign
        replace_sign = [0,0] + des_replace_sign + [0] + attr_replace_sign
        tokens_length = len(all_tokens)
        if tokens_length > self.max_words:
            all_tokens = all_tokens[:self.max_words]
            mask_sign  = mask_sign[:self.max_words]
            replace_sign = replace_sign[:self.max_words]

        pad_length = self.max_words - tokens_length
        tokens_length = len(all_tokens)
        pad_tokens = ['[PAD]'] * pad_length
        replace_sign = replace_sign + [0]*pad_length
        all_tokens = all_tokens + pad_tokens

        mask_token = des_mask_token + attr_mask_token

        mask_token = mask_token[:sum(mask_sign)]

        input_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
        mask_token_ids = self.tokenizer.convert_tokens_to_ids(mask_token)

        mask_labels = [-100] * self.max_words
        mask_pos = [i for i,a in enumerate(mask_sign) if a ==1]

        assert len(mask_pos) == len(mask_token_ids)

        for p,l in zip(mask_pos, mask_token_ids):
            mask_labels[p]=l 
        replace_labels = replace_sign

        attention_mask = [1]* tokens_length + [0] * pad_length

        ####change list to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mask_labels = torch.tensor(mask_labels, dtype=torch.long)
        replace_labels = torch.tensor(replace_labels, dtype=torch.long)

        return input_ids, attention_mask, mask_labels, replace_labels

    def __load_product_info(self):
        indata = json.load(open(self.product_list_filename, mode='r', encoding='utf8'))
        product_idlist = []
        product_indexmap = {}
        for p in indata:
            product_id = p[0]
            product_idlist.append(product_id)
            product_indexmap[product_id] = p[1]
        return product_idlist, product_indexmap

    def __load_capids(self):
        cap_index = {}
        count = 0
        for item in self.info_data:
            product_id = item['input_productID']
            if product_id not in cap_index.keys():
                cap_index[product_id] = count
                count += 1
        return cap_index
    
    def __load_cate_map(self):
        cate_map = {}
        with open(self.catemap_filename, mode='r', encoding='utf8') as mapfile:
            lines = mapfile.readlines()
            for line in lines:
                cate, symbol = line.strip().split('\t')
                cate_map[cate] = symbol
        return cate_map
        
    def __len__(self):
        # return len(self.product_list)
        return len(self.info_data)
    
    # another data feeding way
    # def __getitem__(self, index):  
    #     product_id = self.product_list[index]
    #     cap_index = random.choice(self.product_indexmap[product_id])
    #     ####make sure align
    #     assert product_id == self.raw_data['input_productID'][cap_index][0]
    #     input_ids, attention_mask, mask_labels, replace_labels = self.__get_text(cap_index)
    #     img = self.__get_image(cap_index)
    #     return img, input_ids, attention_mask, mask_labels, replace_labels, cap_index
    
    def __getitem__(self, index):  
        product_id = self.info_data[index]['input_productID']
        input_ids, attention_mask, mask_labels, replace_labels = self.__get_text(index)
        img = self.__get_image(index)
        cap_id = self.cap_ids[product_id]
        return img, input_ids, attention_mask, mask_labels, replace_labels, cap_id

class fashiongen_dataset_retrieval(Dataset):
    def __init__(self, args, transform, tokenizer, split):   
        self.data_root = args.data_root
        # self.sub_dataset = args.sub_dataset
        self.split = 'validation' if split in ('val', 'test') else split
        self.max_words = args.max_word_num
        self.catemap_filename = os.path.join(self.data_root,args.catemap_filename)
        self.product_list_filename = os.path.join(self.data_root, self.split, args.product_list_filename)
        self.transform = transform
        self.info_data, self.raw_data= self.__load_datafile()
        self.product_list, self.product_indexmap = self.__load_product_info()


        self.tokenizer = tokenizer
        if self.split == 'validation':
            self.texts = self.__extract_all_texts()
            self.subcate_map = self.__cluster_subcate()


    def __load_datafile(self):
        datafilename = os.path.join(self.data_root, self.split, 'info_data.json')
        infodata = json.load(open(datafilename, mode='r', encoding='utf-8'))
        raw_datafilename = os.path.join(self.data_root, 'fashiongen_256_256_{}.h5'.format(self.split))
        raw_data = h5py.File(raw_datafilename)
        assert len(infodata) == len(raw_data['index_2'])
        return infodata, raw_data

    def __load_product_info(self):
        indata = json.load(open(self.product_list_filename, mode='r', encoding='utf8'))
        product_idlist = []
        product_indexmap = {}
        for p in indata:
            product_id = p[0]
            product_idlist.append(product_id)
            product_indexmap[product_id] = p[1]
        return product_idlist, product_indexmap

    def __get_image(self, index):
        img = Image.fromarray(self.raw_data['input_image'][index])
        img = self.transform(img)
        return img

    def __get_text(self, index):
        item_info = self.info_data[index]
        ###### process description
        description = 'the image description is ' + item_info['input_description']
        description_tokens = self.tokenizer.tokenize(pre_caption(description))
        description_tokens = ['[CLS]'] + description_tokens
        tokens_length = len(description_tokens)
        if tokens_length > self.max_words:
            description_tokens = description_tokens[:self.max_words]
            tokens_length = self.max_words
        pad_length = self.max_words - tokens_length
        input_ids = self.tokenizer.convert_tokens_to_ids(description_tokens)
        attention_mask = [1]* len(input_ids) + [0] * pad_length
        input_ids = input_ids + [0] * pad_length

        ####change list to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return input_ids, attention_mask
        
    def __len__(self):
        if self.split == 'train':
            return len(self.product_list)
        elif self.split == 'validation':
            return len(self.info_data)
    
    def __getitem__(self, index):
        if self.split == 'train':
            product_id = self.product_list[index]
            cap_index = random.choice(self.product_indexmap[product_id])
            assert product_id == self.raw_data['input_productID'][cap_index][0]
        elif self.split == 'validation':
            cap_index = index
        
        input_ids, attention_mask = self.__get_text(cap_index)
        img = self.__get_image(cap_index)
        return img, input_ids, attention_mask, cap_index

    def __extract_all_texts(self):
        texts = []
        for i,pid in enumerate(self.product_list):
            cap_index = self.product_indexmap[pid][0]
            descripton = self.info_data[cap_index]['input_description']
            texts.append('the image description is ' + pre_caption(descripton))

        return texts

    def get_test_labels(self):
        if self.split == 'train':
            return None
        text_num = len(self.product_list)
        img_num = len(self.raw_data['index'])
        text2img = {i:self.product_indexmap[pid] for i,pid in enumerate(self.product_list)}
        img2text = {}
        for k,v in text2img.items():
            for vv in v:
                img2text[vv] = k
        assert text_num == len(text2img)
        assert img_num == len(img2text)
        return img2text, text2img

    def __cluster_subcate(self):
        subcate_map = {}

        for i, item_info in enumerate(self.info_data):
            subcat = item_info['input_subcategory']
            product_id = item_info['input_productID']
            if subcat in subcate_map:
                subcate_map[subcat].add(product_id)
            else:
                subcate_map[subcat] = set([product_id])
        for k,v in subcate_map.items():
            subcate_map[k] = list(v)
        return subcate_map



    def get_i2t_test(self, set_len=1000, neg_len=100):
        pid_indexes = list(range(len(self.product_list)))
        random.shuffle(pid_indexes)
        pid_indexes = pid_indexes[:set_len]
        img_indexes = []
        pid_to_index = {pid:i for i,pid in enumerate(self.product_list)}
        txt_indexs = []

        for pid_ind in pid_indexes:
            pid = self.product_list[pid_ind]
            img_index = random.choice(self.product_indexmap[pid])
            img_indexes.append(img_index)
            ### find subcate/////
            subcates = list(self.subcate_map.keys())
            random.shuffle(subcates)
            subcate = self.info_data[img_index]['input_subcategory']
            subcates.remove(subcate)
            neg_pids = deepcopy(self.subcate_map[subcate])
            neg_pids.remove(pid)
            while len(neg_pids) < 100:
                subcate = subcates.pop()
                neg_pids.extend(self.subcate_map[subcate])

            neg_pids = random.sample(neg_pids,neg_len)
            neg_indexes = [pid_to_index[pid] for pid in neg_pids]
            txt_indexs.append(neg_indexes+[pid_ind])
        
        img_indexes = np.array(img_indexes)
        txt_indexs = np.array(txt_indexs)
        return img_indexes, txt_indexs

    def get_t2i_test(self, set_len=1000, neg_len = 100):
        pid_indexes = list(range(len(self.product_list)))
        random.shuffle(pid_indexes)
        pid_indexes = pid_indexes[:set_len]
        txt_indexes = []
        pid_to_index = {pid:i for i,pid in enumerate(self.product_list)}
        img_indexs = []
        for pid_ind in pid_indexes:
            pid = self.product_list[pid_ind]
            txt_indexes.append(pid_ind)
            subcates = list(self.subcate_map.keys())
            random.shuffle(subcates)
            pos_img_index = random.choice(self.product_indexmap[pid])
            subcate = self.info_data[pos_img_index]['input_subcategory']
            neg_pids = deepcopy(self.subcate_map[subcate])
            neg_pids.remove(pid)
            while len(neg_pids) < 100:
                subcate = subcates.pop()
                neg_pids.extend(self.subcate_map[subcate])
            neg_pids = random.sample(neg_pids, neg_len)
            neg_indexes = [random.choice(self.product_indexmap[i]) for i in neg_pids]
            img_indexs.append(neg_indexes+[pos_img_index])

        txt_indexes = np.array(txt_indexes)
        img_indexs = np.array(img_indexs)
        return txt_indexes, img_indexs

