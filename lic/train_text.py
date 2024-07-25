# Copyright (c) 2021-2022, InterDigital Communications, Inc
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

import argparse
import random
import shutil
import sys
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from losses.rate_distortion import RateDistortionMultiLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from torchvision.datasets import CocoCaptions
from typing import Any, Tuple

class CustomCOCOCaptionsDataset(CocoCaptions):
    def __init__(self, root, annFile, transforms=None, min_size = [256,256]):
        super().__init__(root, annFile, transforms)
        self.min_size = min_size
    def _check_size(self, img):
        return img.size[0] >= self.min_size[0] and img.size[1] >= self.min_size[1]
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)[0]

        if not self._check_size(image):
            return None, None
        if self.transforms is not None:
            image, target  = self.transforms(image, target)
        return image, target
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data = [item for item in data if item[0] is not None]
    if len(data) == 0:
        return None, None, 0
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    #print("captions", captions)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    #targets = torch.zeros(len(captions), max(lengths)).long()       
    return images, captions

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


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, ds in enumerate(train_dataloader):
        d, txt  = ds
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d, txt)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tLPIPS loss: {out_criterion["lpips_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f} |"
                #f"\tsigma: {model.sigma:.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    lpips_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for ds in test_dataloader:
            d, txt = ds
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d, txt)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            lpips_loss.update(out_criterion["lpips_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tLPIPS loss: {lpips_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, model_info):
    filename= model_info + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_info + "_checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--model",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def _linear_schedule(t, T, sigma_init, sigma_max):
    if t <= T:
        slope = (sigma_max - sigma_init) / T
        return sigma_init + slope * t
    else:
        return sigma_max

def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )
    #TODO: Crop size should be larger than 256. Maybe 512?

    train_dataset = CustomCOCOCaptionsDataset(transforms=train_transforms, 
                                 root="/root/shared_smurai/COCO/train2014", 
                                 annFile="/root/shared_smurai/COCO/annotations/captions_train2014.json") #train dataset path
    test_dataset = CustomCOCOCaptionsDataset(root="/root/shared_smurai/COCO/val2014",
                                transforms=test_transforms,
                                annFile="/root/shared_smurai/COCO/annotations/captions_val2014.json") #test dataset path

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn
    )

    net = image_models[args.model](quality=1)
    net = net.to(device)


    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    metric_type = ['mse', 'lpips-vgg', 'clipi2t', 'clipiqa']
    kappa = {'mse': 0.5, 'lpips-vgg':0.2, 'clipi2t':0.2, 'clipiqa':0.1}
    criterion = RateDistortionMultiLoss(
        lmbda=args.lmbda, kappa = kappa, metric_type=metric_type
       )

    last_epoch = 0
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        #last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint)
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        net.sigma = _linear_schedule(epoch, args.epochs, 5, 10000)
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                str(net.__class__.__name__) + "_" + str(args.lmbda) + "_" + str(kappa)
            )


if __name__ == "__main__":
    main(sys.argv[1:])

