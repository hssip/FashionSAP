
import os
from torch.utils.data import Dataset
import torch

from PIL import Image
import ujson as json
from .utils import pre_caption

'''
fashioniq dataset has unique target list and no-unique candidate list
'''
class fashiongen_dataset_tgir_train(Dataset):
    def __init__(self, args, transform, tokenizer, split):   
        self.data_root = args.data_root
        self.sub_dataset = args.sub_dataset
        self.split = 'val' if split in ('val', 'test') else split
        self.train_class = args.train_class
        self.max_words = args.max_word_num
        self.transform = transform
        self.data_list = self.__load_datafile()
        if self.sub_dataset:
            self.data_list = self.data_list[:200]
        self.tokenizer = tokenizer
        self.img_ids = self.__get_img_ids()


    def __load_datafile(self):
        datafilename = 'cap.train.json' if self.train_class == 'all' else 'cap.{}.train.json'.format(self.train_class)
        infile_path = os.path.join(self.data_root,'captions',datafilename)
        data_list = json.load(open(infile_path, mode='r', encoding='utf-8'))
        return data_list
    
    def __get_img_ids(self):
        img_ids = set()
        for item in self.data_list:
            img_ids.add(item['candidate'])
            img_ids.add(item['target'])
        img_ids = {iid:i for i,iid in enumerate(img_ids)}
        return img_ids

    def __get_image(self, index):
        ref_img_name = self.data_list[index]['candidate']
        tar_img_name = self.data_list[index]['target']
        ref_id = self.img_ids[ref_img_name]
        tar_id = self.img_ids[tar_img_name]

        ref_path = os.path.join(self.data_root,'images', ref_img_name+'.png')
        ref_img = Image.open(ref_path).convert('RGB')
        ref_img = self.transform(ref_img)
        tar_path = os.path.join(self.data_root,'images', tar_img_name+'.png')
        tar_img = Image.open(tar_path).convert('RGB')
        tar_img = self.transform(tar_img)
        return ref_img, tar_img, ref_id, tar_id

    def __get_text(self, index):
        item_info = self.data_list[index]
        ###### process description

        description = 'the difference between two images are ' + ' and '.join(item_info['captions'])
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

        return len(self.data_list)
    
    def __getitem__(self, index):

        input_ids, attention_mask = self.__get_text(index)
        reference_img, target_img, ref_id, tar_id = self.__get_image(index)
        
        return input_ids, attention_mask, reference_img, target_img, ref_id, tar_id, index
    

class fashiongen_dataset_tgir_val(Dataset):
    def __init__(self, args, transform, tokenizer, split, val_class):   
        self.data_root = args.data_root
        self.sub_dataset = args.sub_dataset
        self.split = 'val' if split in ('val', 'test') else split
        self.val_class = val_class
        self.max_words = args.max_word_num
        self.transform = transform
        self.data_list = self.__load_datafile()
        if self.sub_dataset:
            self.data_list = self.data_list[:200]
        self.texts, self.imgs, self.labels = self.__get_imgs_texts()
        self.tokenizer = tokenizer


    def __load_datafile(self):
        datafilename = 'cap.val.json' if self.val_class == 'all' else 'cap.{}.val.json'.format(self.val_class)
        infile_path = os.path.join(self.data_root,'captions',datafilename)
        data_list = json.load(open(infile_path, mode='r', encoding='utf-8'))
        return data_list
    
    def __get_imgs_texts(self):
        texts = []
        imgs_set = set()
        for data in self.data_list:
            imgs_set.add(data['candidate'])
            imgs_set.add(data['target'])
        imgs = list(imgs_set)
        labels = []
        for i, data in enumerate(self.data_list):
            texts.append(pre_caption('the difference between two images are ' + ' and '.join(data['captions'])))
            ref_img_index = imgs.index(data['candidate'])
            tar_img_index = imgs.index(data['target'])
            labels.append((i, ref_img_index, tar_img_index))
            
        return texts, imgs, labels
    
    def __get_single_image(self, index):
        fname = self.imgs[index]
        fpath = os.path.join(self.data_root, 'images',fname+'.png')
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        return img
        

    def __get_pair_images(self, index):
        ref_img_name = self.data_list[index]['candidate']
        tar_img_name = self.data_list[index]['target']

        ref_path = os.path.join(self.data_root,'images', ref_img_name+'.png')
        ref_img = Image.open(ref_path).convert('RGB')
        ref_img = self.transform(ref_img)
        tar_path = os.path.join(self.data_root,'images', tar_img_name+'.png')
        tar_img = Image.open(tar_path).convert('RGB')
        tar_img = self.transform(tar_img)
        return ref_img, tar_img
    

    def __get_text(self, index):
        item_info = self.data_list[index]
        ###### process description
        description = 'the difference between two images are ' + ' and '.join(item_info['captions'])
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
        return len(self.data_list)
    
    def __getitem__(self, index):

        input_ids, attention_mask = self.__get_text(index)
        reference_img, target_img = self.__get_pair_images(index)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return input_ids, attention_mask, reference_img, target_img, label
    
    def get_labels(self):
        return self.labels
    
    
class fashiongen_dataset_tgir_originval(Dataset):
    def __init__(self, args, transform, tokenizer, split):   
        self.data_root = args.data_root
        self.max_words = args.max_word_num
        self.transform = transform
        self.data_list = self.__load_datafile()
        self.texts, self.imgs, self.labels = self.__get_imgs_texts()
        self.tokenizer = tokenizer


    def __load_datafile(self):
        datafilename = 'cap.val.json'
        infile_path = os.path.join(self.data_root,'captions',datafilename)
        data_list = json.load(open(infile_path, mode='r', encoding='utf-8'))
        return data_list
    
    def __get_imgs_texts(self):
        texts = []
        imgs = json.load(open())
        labels = []
        for i, data in enumerate(self.data_list):
            texts.append(pre_caption('the difference between two images is ' + ' and '.join(data['captions'])))
            ref_img_index = imgs.index(data['candidate'])
            tar_img_index = imgs.index(data['target'])
            labels.append((i, ref_img_index, tar_img_index))
            
        return texts, imgs, labels
    
    def __get_single_image(self, index):
        fname = self.imgs[index]
        fpath = os.path.join(self.data_root, 'images',fname+'.png')
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        return img
        

    def __get_pair_images(self, index):
        ref_img_name = self.data_list[index]['candidate']
        tar_img_name = self.data_list[index]['target']

        ref_path = os.path.join(self.data_root,'images', ref_img_name+'.png')
        ref_img = Image.open(ref_path).convert('RGB')
        ref_img = self.transform(ref_img)
        tar_path = os.path.join(self.data_root,'images', tar_img_name+'.png')
        tar_img = Image.open(tar_path).convert('RGB')
        tar_img = self.transform(tar_img)
        return ref_img, tar_img
    

    def __get_text(self, index):
        item_info = self.data_list[index]
        ###### process description
        description = 'the difference between two images is ' + ' and '.join(item_info['captions'])
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
        return len(self.data_list)
    
    def __getitem__(self, index):

        input_ids, attention_mask = self.__get_text(index)
        reference_img, target_img = self.__get_pair_images(index)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return input_ids, attention_mask, reference_img, target_img, label
    
    def get_labels(self):
        return self.labels
    