# ImageTextCoding
A temporal repository for the paper "LMM-driven Semantic Image-Text Coding for Ultra Low-bitrate Image Compression"

# Description 
This is the repository fot the under-review paper "LMM-driven Semantic Image-Text Coding for Ultra Low-bitrate Image Compression".
This repository is under temporal GitHub account to maintain the anonimity. This will be moved to actual repository of the authors later.  

# Requirements
Python 3.10 and some other packages are needed. Please refer to the How to Use section below.
Note that our experiments and verification are conducted on Linux(Ubuntu 22.04).

# How to Use

- First, clone this repository. 
```
git clone https://github.com/user475289/TextImageCoding/
cd TextImageCoding
```

- Download the DiffBIR weights and our pre-trained weights to the `\weights` folder. 
- Install requirements (using virtual environment is recommended).
```
pip install -r requirements.txt
```

## demo inference
We prepare compressed text caption for kodak image datasets. Please just run
```
bash inference.sh
```
to performance comparison. 

For other datasets, please generate and compress the caption by running `llavanextCaption_Compression.ipynb` and place the output csv to the `df` folder, and specify the dataset in `inference.sh`. 

## Training
Our training code is based on CompressAI. Please run `train.sh` with specification of the models, datasets and parameters. 

