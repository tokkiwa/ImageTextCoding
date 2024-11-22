<a href="https://arxiv.org/abs/2411.13033"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2411.13033-red"></a>




# ImageTextCoding
Official repository for the paper "LMM-driven Semantic Image-Text Coding for Ultra Low-bitrate Image Compression" (IEEE VCIP 2024)

![h=20](https://github.com/tokkiwa/ImageTextCoding/blob/main/assets/comparison15.drawio.png)

[Check out our presentation poster!](https://github.com/tokkiwa/ImageTextCoding/blob/main/assets/VCIP_Poster_draft_1122.pdf)

# Description 
This is the official repository for the paper "LMM-driven Semantic Image-Text Coding for Ultra Low-bitrate Image Compression". Full paper is available on [arXiv](https://arxiv.org/abs/2411.13033).

Please feel free to contact Murai(octachoron(at)suou.waseda.jp), [Sun Heming ](https://sun.ynu.ac.jp/) or post an issue if you have any questions. 

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
git clone https://github.com/tokkiwa/TextImageCoding/
cd TextImageCoding
```

- Download the DiffBIR weights and our pre-trained weights to the `/weights` folder and `/lic-weights/cheng` folder respectively.

The weights for DiffBIR is available at https://github.com/XPixelGroup/DiffBIR. 
We adopt 'v1_general' weights through our experiments. 

Our pre-trained weight is avairable at [this link](https://github.com/tokkiwa/ImageTextCoding/releases/download/v0.1-alpha/10ep_cheng_3_mse0.5_vgg0.2_i2t0.2_iqa0.1.tar). Please note that this is nightly release. 
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

## Ackownledgement
Our codes are based on [MISC](https://github.com/lcysyzxdxc/MISC), [CompressAI](https://github.com/InterDigitalInc/CompressAI), [GPTZip](https://github.com/erika-n/GPTzip) and [DiffBIR](https://github.com/XPixelGroup/DiffBIR). We thank the authors for releasing their excellent work. 
