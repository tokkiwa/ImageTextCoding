# ImageTextCoding
A temporal repository for the paper "LMM-driven Semantic Image-Text Coding for Ultra Low-bitrate Image Compression"

![h=20](https://github.com/user475289/ImageTextCoding/blob/main/assets/comparison15.drawio.png)
# Description 
This is the repository fot the under-review paper "LMM-driven Semantic Image-Text Coding for Ultra Low-bitrate Image Compression".
**This GitHub account is tentative to maintain the anonimity.** This repository will be moved to actual repository of the authors later.  

# Demo Inference on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/user475289/ImageTextCoding/blob/master/ImageTextCoding.ipynb)

We prepare demo inference code for google colaboratory.
You can check the inference without any environmental setting. Just click the 'Open in Colab' button above. 

# Requirements
Python 3.10 and some other packages are needed. Please refer to the `How to Use` section below.
Our experiments and verification are conducted on Linux(Ubuntu 22.04) and Docker container with cuda=1.2.1 and torch=2.1. 

# How to Use

- First, clone this repository. 
```
git clone https://github.com/user475289/TextImageCoding/
cd TextImageCoding
```

- Download the DiffBIR weights and our pre-trained weights to the `/weights` folder and `/lic-weights/cheng` folder respectively.

The weights for DiffBIR is available at https://github.com/XPixelGroup/DiffBIR. 
We adopt 'v1_general' weights through our experiments. 

Our pre-trained weight is avairable at [this link](https://github.com/user475289/ImageTextCoding/releases/download/v0.1-alpha/10ep_cheng_3_mse0.5_vgg0.2_i2t0.2_iqa0.1.tar). Please note that this is nightly release. 
All the weights for the experiment will be released soon. 

- Install requirements (using virtual environment is recommended).
```
pip install -r requirements.txt
```
## Caption Generation and Compression
Codes for Caption Generation and Compression can be found in `llavanextCaption_Compression.ipynb`. 

## Inference
We prepare text caption for kodak image datasets. Please run
```
bash run_misc.sh
```
with necessary specification. 

For other datasets, please generate and compress the caption by running `llavanextCaption_Compression.ipynb` and place the output csv to the `df` folder, and specify the dataset in `run_misc.sh`. 

## Training
Our training code is based on CompressAI. Please run `lic/train.sh` with specification of the models, datasets and parameters. 

