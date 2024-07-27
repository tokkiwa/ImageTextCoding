#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_misc.py --dataset clic --lic_model cheng --model_desc cheng_vgg0.5_mse0.5 \
--df df/clic_llava_1.6_vicuna.csv --path lic_weights/cheng/Cheng2020Attention_3.0_mse0.5_vgg0.5_checkpoint_best_loss.pth.tar \
--mode tuned_net --save_path MISC_outputs/CLIC_cheng_vgg0.5_mse0.5 \
--from_encoded MISC_outputs/clic_encoded_cheng_vgg0.5_mse0.5.pickle

