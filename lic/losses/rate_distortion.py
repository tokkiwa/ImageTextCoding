# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import torch
import torch.nn as nn

from pytorch_msssim import ms_ssim

from compressai.registry import register_criterion
from lpips import LPIPS
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
from torchmetrics.multimodal.clip_score import CLIPScore
import clip

class CLIPCosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        device = "cuda" if torch.cuda.is_available else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
    
    def forward(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)
        x_features = self.model.encode_image(x)
        y_features = self.model.encode_image(y)
        return torch.nn.functional.cosine_similarity(x_features, y_features)
        
        


def scale4LPIPS(images: torch.Tensor) -> torch.Tensor:
    return images * 2 - 1
class RateDistortionMultiLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""


    def __init__(self, lmbda= 3, kappa = {'mse': 0.5, 'lpips-alex':0.2, 'clipsim_i2t': 0.3},  metric_type=["mse", "lpips-alex", "clipsim_i2t"], return_type="all"):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available else "cpu"
        #device = "cpu"
        
        self.metric_type = metric_type
        if "mse" in metric_type:
            self.metric_mse = nn.MSELoss()
        if "ms-ssim" in metric_type:
            self.metric_msssim = ms_ssim
        if "lpips-alex" in metric_type:
            self.metric_lpips_alex = LPIPS(net='alex').to(device)
        if "lpips-vgg" in metric_type:
            self.metric_lpips_vgg = LPIPS(net='vgg').to(device)
        if "clipiqa" in metric_type:
            self.metric_clipiqa = CLIPImageQualityAssessment().to(device)
        if "clipsim_i2i" in metric_type:
            self.metric_clipsim_i2i = CLIPCosineSimilarity().to(device)
        if "clipsim_i2t"in metric_type:
            self.metric_clipsim_i2t = CLIPScore(model_name_or_path = 'openai/clip-vit-base-patch16',
                                                ).to(device)
            
        self.lmbda = lmbda
        self.kappa = kappa
        self.return_type = return_type

    def forward(self, output, target, txt = None, **kwargs):
        #print("output is in [",  output["x_hat"].min(), output["x_hat"].max(), "]")
        #print("target is in [",  target.min(), target.max(), "]")
        
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        sum_distortion = 0
        if "ms_ssim" in self.metric_type:
            out["ms_ssim_loss"] = self.metric_msssim(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
            sum_distortion += self.kappa["ms_ssim"] * distortion
        if  "mse" in self.metric_type:
            out["mse_loss"] = self.metric_mse(output["x_hat"], target)
            distortion = out["mse_loss"]
            sum_distortion += self.kappa["mse"] * distortion
        if  "lpips-alex" in self.metric_type:
            out["lpips_loss"] = self.metric_lpips_alex(scale4LPIPS(output["x_hat"]), scale4LPIPS(target)).squeeze().mean()
            distortion = out["lpips_loss"]
            sum_distortion += self.kappa["lpips-alex"] * distortion
        if  "lpips-vgg" in self.metric_type:
            out["lpips_loss"] = self.metric_lpips_vgg(scale4LPIPS(output["x_hat"]), scale4LPIPS(target)).squeeze().mean()
            distortion = out["lpips_loss"]
            sum_distortion += self.kappa["lpips-vgg"] * distortion
        if  "clipiqa" in self.metric_type:
            out["clipiqa_loss"] = self.metric_clipiqa(output["x_hat"]).mean()
            distortion = out["clipiqa_loss"]
            sum_distortion += self.kappa["lpips-vgg"] * distortion
            
        if  "clipsim_i2i" in self.metric_type:
            out["clipsim_i2i_loss"] = self.metric_clipsim_i2i(output["x_hat"], target).squeeze().mean()
            distortion = 1 - out["clipsim_i2i_loss"]
            sum_distortion += self.kappa["clipsim_i2i"] * distortion
            
        if  "clipsim_i2t" in self.metric_type:
            out["clipsim_i2t_loss"] = self.metric_clipsim_i2t(output["x_hat"].clamp(0,1), txt).squeeze().mean()
            distortion = 1 - out["clipsim_i2t_loss"] / 100.
            sum_distortion += self.kappa["clipsim_i2t"] * distortion           
        out["loss"] = self.lmbda * sum_distortion + out["bpp_loss"]
        #print("loss:", out["loss"])
        #print(out["loss"].shape)
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
