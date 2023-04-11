## FashionSAP: Symbols and Attributes Prompt for Fine-grained Fashion Vision-Language Pre-training

CVPR2023.

This is the source code of PyTorch implementation of the FashionSAP. 
We will introduce more about this project ...

### Requirements:
* requirements.txt

### Prepare:

* FashionGen [link](https://arxiv.org/abs/1806.08317)

    1. download the raw file and extract it in path `data_root`.
    2. change the `data_root` and `split` in `prepare_dataset.py` and run it get the assitance file.

* Fashioniq [link](https://arxiv.org/abs/1905.12794)
    
    1. download the raw file and extract it in path `data_root`.
    2. the directory `captions` and `images` in raw fileare put in `data_root`. Besides the file, we also merge all kinds of train file into `cap.train.json` file in `captions`, so as to `val`.

### Run


### Cite

