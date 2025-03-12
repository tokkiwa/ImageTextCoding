#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_misc.py --dataset kodak --lic_model cheng --model_desc cheng_tuned \
--df kodak_llava_1.5.csv --path lic-weights/cheng/cheng_1_mse0.5_vgg0.2_i2t0.2_iqa0.1.pth.tar \
--mode tuned_net --save_path outputs/kodak_llava

