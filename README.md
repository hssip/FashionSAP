## FashionSAP: Symbols and Attributes Prompt for Fine-grained Fashion Vision-Language Pre-training

This is the source code of PyTorch implementation of the FashionSAP:. 

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
1. we define 3 downstream names as `downstream_name`

    * `retrieval`: text-to-image retrieval and image-to-text retrieval 
    * `catereg`: fashion domain category recognition and subcategory recognition
    * `tgir`:  text guided image retrieval or text modified image retrieval

1. command `bash run_pretrain.sh` to run pretrain stage
2. command `bash run_{downstream_name}.sh` to train and evaluate different downstream tasks


