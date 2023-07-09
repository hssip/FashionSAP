## FashionSAP: Symbols and Attributes Prompt for Fine-grained Fashion Vision-Language Pre-training

This paper is accepted by the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2023 (CVPR2023) [paper](https://arxiv.org/abs/2304.05051)

This is the source code of PyTorch implementation of the FashionSAP. 

We will introduce more about our project ...

### Requirements:
* requirements.txt

### Prepare:

* [FashionGen](https://arxiv.org/abs/1806.08317)

    1. download the raw file and extract it in path `data_root`.
    2. change the `data_root` and `split` in `prepare_dataset.py` and run it get the assitance file.

* [Fashioniq](https://arxiv.org/abs/1905.12794)
    
    1. download the raw file and extract it in path `data_root`.
    2. the directory `captions` and `images` in raw fileare put in `data_root`. Besides the file, we also merge all kinds of train file into `cap.train.json` file in `captions`, so as to `val`.

### Run
1. we define 3 kinds downstream names as `downstream_name`

    * `retrieval`: includes 2 downstream tasks: text-to-image retrieval downstream and image-to-text retrieval 
    * `catereg`: fashion domain category recognition and subcategory recognition
    * `tgir`:  text guided image retrieval or text modified image retrieval

1. command `bash run_pretrain.sh` to run pretrain stage, we also public our pre-trained model weight file in [Google Driver](https://drive.google.com/file/d/1yKm5b1L5rCe6HT7FkQ1Ta5D2HNstc5nv/view?usp=sharing)
2. command `bash run_{downstream_name}.sh` to train and evaluate different downstream tasks

### Citations
If you find this code useful for your research, please cite:
```
@inproceedings{FashionSAP,
      title={FashionSAP: Symbols and Attributes Prompt for Fine-grained Fashion Vision-Language Pre-training}, 
      author={Han, Yunpeng and Zhang, Lisai and Chen, Qingcai and Chen, Zhijian and Li, Zhonghua and Yang, Jianxin and Cao, Zhao},
      year={2023},
      booktitle={CVPR}
}
```

Some utils codes are referenced from project [ALBEF](https://github.com/salesforce/ALBEF)


