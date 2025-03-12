# -*- coding: utf-8 -*-

from torchvision.extension import *
import math
import io
import torch
from torchvision import transforms
from PIL import Image, ImageChops
import sys
import os
from clip import clip
import torch
import cv2
import numpy as np
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from lpips import LPIPS
from skimage.transform import resize
from utils.image import wavelet_reconstruction
import time
import pandas as pd
import numpy as np
import pickle

import shutil

rootdir = "." 

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def clear_directory(directory_path):
# Recurrently delete specified directory
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if os.path.isdir(file_path):

            for root, dirs, files in os.walk(file_path):
                for file in files:
                    os.remove(os.path.join(root, file))



def to_block(img,grid=32,level=8):
# 画像を決定的にダウンサンプルする
    g_w=int(img.size[0]/grid)
    g_h=int(img.size[1]/grid)# 縦横とも 1/grid 倍する
    img_resize=img.resize((g_w, g_h))
    img_np=np.floor(np.array(img_resize)/level)*level
    img_np=img_np.astype (np.uint8)
    img_reference = Image.fromarray(img_np).resize(img.size)
    return img_reference, g_w, g_h


def divide_integer(num, n):
    quotient = num // n  # 整数除法，计算商
    remainder = num % n  # 取余数
    result = [quotient] * n  # 创建一个包含n个quotient的列表

    # 将余数平均分配给前几个数
    for i in range(remainder):
        result[i] += 1
    return result

def mask_block(mask,num=8,level=0.35):
    tmp=resize(mask, (num, num), mode='reflect')
    tmp[tmp>level]=255
    tmp[tmp<=level]=0
    rp_mat_0=np.array(divide_integer(mask.shape[0], num),dtype='int')
    rp_mat_1=np.array(divide_integer(mask.shape[1], num),dtype='int')
    return tmp.repeat(rp_mat_1,axis=1).repeat(rp_mat_0,axis=0)

def image_paddle_in(image, num=32):
    # 计算扩充后的宽度和高度
    new_width = ((image.width-1) // num + 1) * num
    new_height = ((image.height-1) // num + 1) * num

    # 创建一个新的扩充后的图像，用空值填充
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # 将原始图像粘贴到扩充后的图像左上角
    new_image.paste(image, (0, 0))
    return new_image,image.width,image.height

def image_paddle_out(image, old_width, old_height):

    return image.crop((0,0,old_width,old_height))

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def clip_map(img,texts,mask_num=8):
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # CLIP architecture surgery acts on the image encoder
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image_features = seg_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(seg_model, texts, device)

        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(seg_model, [""], device)

        # Apply feature surgery for single text
        similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

        mask_0=(similarity_map[0,:,:,0].cpu().numpy() * 255).astype('uint8')
        mask_1=(similarity_map[0,:,:,1].cpu().numpy() * 255).astype('uint8')
        mask_2=(similarity_map[0,:,:,2].cpu().numpy() * 255).astype('uint8')
        mask_0=Image.fromarray(mask_block(mask_0,num=mask_num))
        mask_1=Image.fromarray(mask_block(mask_1,num=mask_num))
        mask_2=Image.fromarray(mask_block(mask_2,num=mask_num))
        return mask_0,mask_1,mask_2

def sr_pipe(img_reference,positive_prompt="",cfg=1.0,steps=40,res=512, cond_scale = 1.0, old_size = None):
    control_img = img_reference
    sr_scale = 1 #超解像の拡大率
    num_samples = 1 #複数枚サンプリングする。しないので1。
    #image_size = old_size
    disable_preprocess_model= False
    strength = 1.0
    cond_scale = 1.0
    use_color_fix = True
    keep_original_size = False
    negative_prompt="Blurry, Low Quality, featureless, too flat"
    sampler = SpacedSampler(model, var_type="fixed_small")

    if sr_scale != 1:
        control_img = control_img.resize(
            tuple(math.ceil(x * sr_scale) for x in control_img.size),
            Image.BICUBIC
        )
    #input size は入力サイズで、これを指定されたimage_sizeにリサイズしてから超解像を行う
    input_size = control_img.size
    #control_img = auto_resize(control_img, image_size)
    h, w = control_img.height, control_img.width
    control_img = pad(np.array(control_img), scale=64) # HWC, RGB, [0, 255]
    control_imgs = [control_img] * num_samples
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    height, width = control.size(-2), control.size(-1)
    cond = {
        "c_latent": [model.apply_condition_encoder(control)],
        "c_crossattn": [model.get_learned_conditioning([positive_prompt] * num_samples)]
    }
    uncond = {
        "c_latent": [model.apply_condition_encoder(control)],
        "c_crossattn": [model.get_learned_conditioning([negative_prompt] * num_samples)]
    }
    model.control_scales = [strength] * 13

    shape = (num_samples, 4, height // 8, width // 8)
    print(f"latent shape = {shape}")
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    samples = sampler.sample(
        steps, shape, cond,
        unconditional_guidance_scale=cond_scale,
        unconditional_conditioning=uncond,
        cond_fn=None, x_T=x_T
    )
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)

    # apply color correction
    if use_color_fix:
        x_samples = wavelet_reconstruction(x_samples, control)

    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    preds = []
    for img in x_samples:
        if keep_original_size:
            # remove padding and resize to input size
            img = Image.fromarray(img[:h, :w, :]).resize(input_size, Image.LANCZOS)
            preds.append(np.array(img))
        else:
            # remove padding
            preds.append(img[:h, :w, :])
    return preds



def encoder(img, name_0, name_1, name_2, detail_0, detail_1, detail_2, detail_all, mode, using_map):
  if using_map:
      mask_0,mask_1,mask_2=clip_map(img,[name_0,name_1,name_2],mask_num)
  else:
      mask_0, mask_1, mask_2 = None, None, None
#reference
  if mode=='pixel':
    old_width, old_height=img.size
    block_num=max(int(max(old_width, old_height)/16),block_num_min)
    img_reference=to_block(img,block_num,2**block_level)

    b_image=block_level*block_num**2
  elif mode=='net' or mode =='tuned_net':
    img, old_width, old_height = image_paddle_in(img, 64)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out_net = comp_net.forward(x)
    out_net['x_hat'].clamp_(0, 1)
    img_reference = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())
    img_reference = image_paddle_out(img_reference, old_width, old_height)
    b_image=compute_bpp(out_net)*img.size[0]*img.size[1]
  elif mode=='ref':
    old_width, old_height=img.size
    img_reference = Image.open(ref_path).convert('RGB')
    b_image=ref_bpp*img.size[0]*img.size[1]
  return {"mask_0": mask_0,
          "mask_1":mask_1,
          "mask_2":mask_2,
          "img_reference":img_reference,
          "b_image": b_image,
          "old_width": old_width,
          "old_height": old_height
          }

def encode_multiple(image_names, images, df, mode, using_map):
  output = {}
  sum_bpp = 0
  for image_name in image_names:
    name_0 = df.loc[image_name, 'item1']
    name_1 = df.loc[image_name, 'item2']
    name_2 = df.loc[image_name, 'item3']
    detail_0 = df.loc[image_name, 'item1_description']
    detail_1 = df.loc[image_name, 'item2_description']
    detail_2 = df.loc[image_name, 'item3_description']
    detail_all = df.loc[image_name, 'overall_description']
    img = images[image_name]
    out_enc = encoder(img, name_0, name_1, name_2, detail_0, detail_1, detail_2, detail_all, mode, using_map)
    output[image_name] = out_enc
    image_bit = out_enc["b_image"]
    text_bit = len(detail_all) * 8
    bpp = (image_bit + text_bit) / (img.size[0] * img.size[1])
    sum_bpp += bpp
    print(f"Encoded: {image_name}. image_bit = {image_bit}, text_bit = {text_bit}, bpp = {bpp}" )
  print(f"Done. Mean bpp = {sum_bpp / len(images)}")
  return output





"""# Prepare for Decoder Model"""

from utils.image import auto_resize, pad
import einops
from utils.common import instantiate_from_config, load_state_dict
from omegaconf import OmegaConf
from model.cldm import ControlLDM
from model.spaced_sampler import SpacedSampler


def decoder(img_reference, detail_0, detail_1, detail_2, detail_all, b_image, old_width, old_height, mask_0, mask_1, mask_2, using_map, steps = 40):
  ########################## Decoder ##########################
    num_inference_steps=40
    exag=1024/max(img_reference.size)
    height=int(img_reference.size[1]*exag/8)*8
    width=int(img_reference.size[0]*exag/8)*8

    #img_reference=img_reference.resize([width,height])
    image = img_reference

    if using_map:
        mask_0=mask_0.resize([width,height])
        mask_1=mask_1.resize([width,height])
        mask_2=mask_2.resize([width,height])
        mask_all=Image.new("RGB", img_reference.size, (255, 255, 255))
        b_mask=mask_num*mask_num*3
        b_word=(len(detail_0)+len(detail_1)+len(detail_2)+len(detail_all))*8
        bpp=(b_image+b_mask+b_word)/(old_width*old_height)
        print('bpp='+str(bpp))

        image_tmp = sr_pipe(image,positive_prompt=detail_0,cfg=3.5,steps=3,res=res)
        image = ImageChops.add(ImageChops.multiply(image_tmp,mask_0.convert("RGB")),
                              ImageChops.multiply(image,Image.fromarray(255-np.array(mask_0)).convert("RGB"))
                              ).resize((old_width, old_height))
    #    image.resize((old_width, old_height)).save(output_folder+'Mask0/'+image_name)

        image_tmp = sr_pipe(image,positive_prompt=detail_1,cfg=3.5,steps=3,res=res)
        image = ImageChops.add(ImageChops.multiply(image_tmp,mask_1.convert("RGB")),
                              ImageChops.multiply(image,Image.fromarray(255-np.array(mask_1)).convert("RGB"))
                              ).resize((old_width, old_height))
    #    image.resize((old_width, old_height)).save(output_folder+'Mask1/'+image_name)

        image_tmp = sr_pipe(image,positive_prompt=detail_2,cfg=3.5,steps=3,res=res)
        image = ImageChops.add(ImageChops.multiply(image_tmp,mask_2.convert("RGB")),
                              ImageChops.multiply(image,Image.fromarray(255-np.array(mask_2)).convert("RGB"))
                              ).resize((old_width, old_height))
    #    image.resize((old_width, old_height)).save(output_folder+'Mask2/'+image_name)

        image = sr_pipe(image,positive_prompt=detail_all,cfg=7,steps=steps,res=res).resize((old_width, old_height))
    #    image.resize((old_width, old_height)).save(output_folder+'SR/'+image_name)

    else:
        b_word=(len(detail_all))*8
        bpp=(b_image+b_word)/(old_height* old_width)
        print('image bit = ', b_image, 'text bit = ', b_word, 'bpp=', str(bpp))

        image = sr_pipe(image,positive_prompt=detail_all,cfg=7,steps=steps,res=res,
                        old_size = (old_width, old_height))
        #output_image = image.resize((old_width, old_height))
    return {"image": image,
            "b_word": b_word,
            "b_image": b_image,
            "pixels": old_height* old_width,
            "bpp": bpp}


def psnr(img0, img1):
    mse = np.mean((img0 - img1) ** 2)
    return 10 * np.log10(255 ** 2 / mse)

def lpips(img0, img1, loss_fn_alex):
    # Variables im0, im1 is a PyTorch Tensor/Variable with shape Nx3xHxW
    # (N patches of size HxW, RGB images scaled in [-1,+1])
    img0 = (TF.to_tensor(img0)/255 - 0.5) * 2
    img0 = img0.unsqueeze(0).to(device)

    img1 = (TF.to_tensor(img1)/255 - 0.5) * 2
    img1 = img1.unsqueeze(0).to(device)
    # Higher means further/more different. Lower means more similar.
    return loss_fn_alex(img0, img1).item()

def decode_multiple(image_names, df, output, using_map, deblurred_images, save_path):
    decoded_image_paths = os.listdir(save_path)
    sum_psnr = 0
    sum_lpips = 0
    sum_bpp = 0
    psnr_meter = AverageMeter()
    lpips_meter = AverageMeter()
    bpp_meter = AverageMeter()
    for image_name in image_names:
        print(f"decoding {image_name}")
        info = output[image_name]
        mask_0=info["mask_0"]
        mask_1=info["mask_1"]
        mask_2=info["mask_2"]
        img_reference=info["img_reference"]
        b_image=info["b_image"]
        old_width=info["old_width"]
        old_height=info["old_height"]
        detail_0 = df.loc[image_name, 'item1_description']
        detail_1 = df.loc[image_name, 'item2_description']
        detail_2 = df.loc[image_name, 'item3_description']
        detail_all = df.loc[image_name, 'overall_description']
        print(detail_all)
        if using_map:
            b_word=(len(detail_0)+len(detail_1)+len(detail_2)+len(detail_all))*8
        else:
            b_word=(len(detail_all))*8
        
        if f"{image_name}" in decoded_image_paths :
            print(f"{image_name} already decoded")
            decoded_img_PIL = Image.open(f"{save_path}/{image_name}").convert('RGB')
            bpp = (info["b_image"] + b_word) / (info["old_width"] * info["old_height"])
            decoded = {"image": decoded_img_PIL, "b_word": b_word, "b_image": info["b_image"], "pixels": info["old_height"] * info["old_width"], "bpp": bpp}
        else: 
            decoded = decoder(info["img_reference"], detail_0, detail_1, detail_2, detail_all,
                        info["b_image"], info["old_width"], info["old_height"], info["mask_0"], info["mask_1"], info["mask_2"], using_map, steps = 40)
            transforms.ToPILImage()(decoded["image"][0]).save(f"{save_path}/{image_name}")
            bpp = decoded["bpp"]
            
        decoded_img = np.array(decoded["image"], dtype = np.float32).squeeze()

        if args.dataset == "clic":
            image_path=f'{rootdir}/CLIC2020/test/{image_name}'
        elif args.dataset == "kodak":
            image_path=f'{rootdir}/kodak/{image_name}'
        orig_img =  np.array(Image.open(image_path).convert('RGB'), dtype = np.float32).squeeze()
        bpp_meter.update(bpp)
        psnr_val = psnr(orig_img, decoded_img)
        psnr_meter.update(psnr_val)
        lpips_val = lpips(orig_img, decoded_img, loss_fn_alex)
        lpips_meter.update(lpips_val)

        print(f"{image_name} decoded; bpp: {bpp}, Total: {len(deblurred_images)}, PSNR: {psnr_val}, LPIPS: {lpips_val}")
        print(f"Mean PSNR: {psnr_meter.avg}, Mean LPIPS: {lpips_meter.avg}, Mean bpp: {bpp_meter.avg}")
        deblurred_images.append(decoded)
    print(f"done. Mean PSNR: {psnr_meter.avg}, Mean LPIPS = {lpips_meter.avg}, Mean bpp =  {bpp_meter.avg}" )


import argparse
device = "cuda"
model: ControlLDM = instantiate_from_config(OmegaConf.load('./configs/model/cldm.yaml'))
ckpt_swinir='./weights/general_full_v1.ckpt'
load_state_dict(model, torch.load(ckpt_swinir, map_location="cpu"), strict=True)
model.freeze()
model.to(device)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, default="kodak", choices=["kodak", "clic", "clic_subset"])
  parser.add_argument("--lic_model", type=str, default="cheng", choices=["cheng", "TCM"])
  parser.add_argument("--model_desc", type=str, default="20ep_Cheng2020Attention_0.0001_1.5")
  parser.add_argument("--df", type=str, default=f"{rootdir}/df/gpt.csv")
  parser.add_argument("--path", type=str, default=f'{rootdir}/lic_weights/cheng/20ep_Cheng2020Attention_0.0001_1.5_checkpoint.pth.tar')
  parser.add_argument("--mode", type=str, default="tuned_net", choices=["net", "tuned_net", "ref", "pixel"])
  parser.add_argument("--using_map", type=bool, default=False)
  parser.add_argument("--save_path", type=str, default=None)
  parser.add_argument("--from_encoded", type=str, default=None)
  parser.add_argument("--skip_lpips", type=bool, default=True)
  args = parser.parse_args()
  mask_num=8
  res=1024
  using_map=args.using_map
  #Load CLIP model
  if using_map:
    print("Loading CLIP Model...")
    BICUBIC = InterpolationMode.BICUBIC
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
      Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    seg_model, preprocess = clip.load("CS-ViT-B/16", device=device)
    #NOTE: This model is in CLIP-surgery. Not included in clip-openai.
    seg_model.eval()
  mode=args.mode

  Images = {}
  df = pd.read_csv(args.df)
  df = df.set_index("image_name")
  # Load Encoder Model
  if args.from_encoded == None:
    print("Loading encoder model...")
    if args.mode=='net':
        from compressai.zoo import cheng2020_attn
        comp_net = cheng2020_attn(pretrained=True, quality = 1).to(device)
    elif args.mode=='tuned_net':
        if args.lic_model == "cheng":
            checkpoint_path = args.path
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint
            for key in ["network", "state_dict", "model_state_dict"]:
                if key in checkpoint:
                    state_dict = checkpoint[key]
            arch='cheng2020-attn'
            from compressai.zoo.image import model_architectures as architectures
            model_cls = architectures[arch]
            comp_net = model_cls.from_state_dict(state_dict).eval().to(device)
        elif args.lic_model == "TCM":
            from lic import TCM
            checkpoint_path = args.path
            comp_net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
            comp_net = comp_net.to(device)
            comp_net.eval()
            dictory = {}
            checkpoint = torch.load(checkpoint_path, map_location=device)
            for k, v in checkpoint["state_dict"].items():
                dictory[k.replace("module.", "")] = v
            comp_net.load_state_dict(dictory)
    elif mode=='ref':
        ref_path='./ref/example-reference.png'
        ref_bpp=0.0421
    elif mode=='pixel':
        block_level=3
        block_num_min=32
    print("Encoder model loaded.")

    
    if args.dataset =="kodak":
        kodak_path = f"{rootdir}/kodak/"
        Image_names = os.listdir(kodak_path)
        for image_name in Image_names:
            image_path = kodak_path + image_name
            img = Image.open(image_path).convert('RGB')
            Images[image_name] = img

    elif args.dataset =="clic":
        clic_path = f"{rootdir}/CLIC2020/test/"
        Image_names = os.listdir(clic_path)
        for image_name in Image_names:
            image_path=clic_path + image_name
            img = Image.open(image_path).convert('RGB')
            Images[image_name] = img

    output = encode_multiple(Image_names, Images, df, mode = "tuned_net", using_map = args.using_map)
    print("Encoding done.")
    with open(f'{rootdir}/outputs/{args.dataset}_encoded_{args.model_desc}.pickle', mode='wb') as fo:
        pickle.dump(output, fo)
    print(f"Saved encoded images to {rootdir}/outputs/{args.dataset}_encoded_{args.model_desc}.pickle")
  else:
      with open(args.from_encoded, 'rb') as f:
          output = pickle.load(f)
          Image_names = list(output.keys())
          print(f"Loaded {len(Image_names)} images")

  loss_fn_alex = LPIPS(net='vgg').to(device)
  # Decode Parts

  deblurred_images = []

  start = time.time()
  
  #create save path directory if not exist
  
  os.makedirs(args.save_path, exist_ok=True)
  
  decode_multiple(Image_names, df, output, using_map = False, deblurred_images = deblurred_images, save_path = args.save_path)

  end = time.time()
  print(f"time: {end-start}")

  output_data = {"deblurred_images": deblurred_images}

  with open(f'{rootdir}/outputs/{args.dataset}_output_{args.model_desc}.pickle', mode='wb') as fo:
    pickle.dump(output_data, fo)