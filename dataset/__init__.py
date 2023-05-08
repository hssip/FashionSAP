

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.fashiongen_dataset import fashiongen_dataset_pretrain, fashiongen_dataset_retrieval
from dataset.fashiongen_dataset_catereg import fashiongen_dataset_catereg
from dataset.fashioniq_dataset_tgir import fashiongen_dataset_tgir_train, fashiongen_dataset_tgir_val



from dataset.randaugment import RandomAugment

def create_dataset(dataset, config, args, tokenizer):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))   
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
       
    if dataset == 'pretrain':
        train_dataset = fashiongen_dataset_pretrain(args, transform=train_transform, tokenizer=tokenizer, split='train')
        test_dataset = fashiongen_dataset_retrieval(args, transform=test_transform, tokenizer=tokenizer, split='validation')
        return train_dataset, test_dataset
    elif dataset == 'retrieval':
        train_dataset = fashiongen_dataset_retrieval(args, transform=train_transform, tokenizer=tokenizer, split='train')
        test_dataset = fashiongen_dataset_retrieval(args, transform=test_transform, tokenizer=tokenizer, split='validation')
        return train_dataset, test_dataset
    elif dataset == 'catereg':
        train_dataset = fashiongen_dataset_catereg(args, transform=train_transform, tokenizer=tokenizer, split='train')
        test_dataset = fashiongen_dataset_catereg(args, transform=test_transform, tokenizer=tokenizer, split='validation')
        return train_dataset, test_dataset
    elif dataset == 'tgir':
        train_dataset = fashiongen_dataset_tgir_train(args, transform=train_transform, tokenizer=tokenizer, split='train')
        dress_test_dataset = fashiongen_dataset_tgir_val(args, transform=test_transform, tokenizer=tokenizer, split='val', val_class='dress')
        toptee_test_dataset = fashiongen_dataset_tgir_val(args, transform=test_transform, tokenizer=tokenizer, split='val', val_class='toptee')
        shirt_test_dataset = fashiongen_dataset_tgir_val(args, transform=test_transform, tokenizer=tokenizer, split='val', val_class='shirt')
        all_test_dataset = fashiongen_dataset_tgir_val(args, transform=test_transform, tokenizer=tokenizer, split='val', val_class='all')
        return train_dataset, dress_test_dataset, toptee_test_dataset, shirt_test_dataset, all_test_dataset



def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    