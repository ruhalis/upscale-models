import os
import random
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.decomposition import PCA

from typing import List, Tuple
import cv2

import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips, math, matplotlib.pyplot as plt
from pathlib import Path

# Real-ESRGAN imports
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class ResBlock(nn.Module):
    """Two 3×3 convolutions + residual scaling (EDSR style)."""
    def __init__(self, channels: int, scale: float = 0.2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.act(self.conv1(x))) * self.scale
        return x + res


class ResDecoder(nn.Module):
    """Upscaler with 2× PixelShuffle stages (overall ×4)."""
    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_blocks: int = 16):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(base_channels) for _ in range(n_blocks)])
        self.up1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.tail = nn.Conv2d(base_channels, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.tail(x)
        return torch.tanh(x)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers: Tuple[int, ...] = (2, 7, 16), layer_weights: List[float] | None = None):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg_partial = nn.Sequential(*[vgg[i] for i in range(max(layers) + 1)])
        self.layers = layers
        if layer_weights is None:
            layer_weights = [1.0] * len(layers)
        self.register_buffer("layer_weights", torch.tensor(layer_weights).view(-1, 1, 1, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x.add(1).div(2) - self.mean) / self.std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_n = self._preprocess(sr)
        hr_n = self._preprocess(hr)
        feats_sr, feats_hr = [], []
        x_sr, x_hr = sr_n, hr_n
        for i, layer in enumerate(self.vgg_partial):
            x_sr = layer(x_sr)
            x_hr = layer(x_hr)
            if i in self.layers:
                feats_sr.append(x_sr)
                feats_hr.append(x_hr)
        loss = 0.0
        for w, fs, fh in zip(self.layer_weights, feats_sr, feats_hr):
            if fs.shape[-2:] != fh.shape[-2:]:
                fs = F.interpolate(fs, size=fh.shape[-2:], mode="bilinear", align_corners=False)
            loss += w * F.l1_loss(fs, fh)
        return loss
class SRLoss(nn.Module):
    """Returns **tuple** → (l1, perceptual, total).  Inputs expected in (‑1,1)."""
    def __init__(self, λ_pix: float = 1.0, λ_perc: float = 0.15):
        super().__init__()
        self.λ_pix, self.λ_perc = λ_pix, λ_perc
        self.perc_loss = VGGPerceptualLoss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor):
        l1   = F.l1_loss(sr, hr)
        perc = self.perc_loss(sr, hr)
        total = self.λ_pix * l1 + self.λ_perc * perc
        return l1, perc, total
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = ResDecoder().to(device)
criterion = SRLoss(λ_pix=1.0, λ_perc=0.2).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=8e-4,                   # ← Unicode hyphen here
    betas=(0.9, 0.999),
    weight_decay=5e-5          # ← And here too
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50, verbose=True)




# -----------------------
# 1) Configuration
# -----------------------
# Your SR epoch to compare
epoch_to_compare = 450
sr_weights = f"weights/SR4_epoch_{epoch_to_compare}.pth"
# Real-ESRGAN weights
finetuned_weights = "experiments/finetune_RealESRGANx4plus_400k/models/net_g_3000.pth"
anime_weights     = "weights/RealESRGAN_x4plus_anime_6B.pth"

# Image list
image_paths = [
    "dataset/assault_lily_bouquet_1.png",
    "dataset/violet_evergarden_0.png",
    "dataset/magia_record_s1_0.png"
]

crop_size    = 512
low_res_size = 128
scale_factor = 4




model.load_state_dict(torch.load(sr_weights, map_location=device))
model.eval()

# -----------------------
# 3) Setup Real-ESRGAN upsamplers
# -----------------------
# Fine-tuned ESRGAN
model_ft = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale_factor)
upsampler_ft = RealESRGANer(
    scale=scale_factor,
    model_path=finetuned_weights,
    dni_weight=None,
    model=model_ft,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=0 if torch.cuda.is_available() else None
)
# Anime ESRGAN
model_an = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale_factor)
upsampler_an = RealESRGANer(
    scale=scale_factor,
    model_path=anime_weights,
    dni_weight=None,
    model=model_an,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=0 if torch.cuda.is_available() else None
)

# -----------------------
# 4) Image processing
# -----------------------
def process_image(img: Image.Image, crop_size=512, low_res_size=128):
    w, h = img.size
    if min(w, h) < crop_size:
        scale = crop_size / min(w, h) * 1.2
        img   = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        w, h  = img.size

    left = (w - crop_size)//2
    top  = (h - crop_size)//2
    crop = img.crop((left, top, left+crop_size, top+crop_size))
    lr   = crop.resize((low_res_size, low_res_size), Image.BICUBIC)

    to_tensor = T.ToTensor()
    hr_t = to_tensor(crop).unsqueeze(0).to(device) * 2.0 - 1.0
    lr_t = to_tensor(lr).unsqueeze(0).to(device) * 2.0 - 1.0
    return lr, hr_t, lr_t

# -----------------------
# 5) Setup plot grid
# -----------------------
n_images = len(image_paths)
n_cols   = 3 + 1  # sr_epoch, ft, anime, original
fig = plt.figure(figsize=(4*n_cols, 4*n_images))
gs  = GridSpec(n_images, n_cols, figure=fig, wspace=0.05, hspace=0.05)

col_labels = [
    f"SR Epoch {epoch_to_compare}",
    "Real-ESRGAN\nFine-tuned",
    "Real-ESRGAN\nAnime",
    "Original\nHigh-res"
]
for c, label in enumerate(col_labels):
    ax = fig.add_subplot(gs[0, c])
    ax.set_title(label, pad=8)
    ax.axis('off')

# -----------------------
# 6) Inference & plotting
# -----------------------
for r, img_path in enumerate(image_paths):
    img = Image.open(img_path).convert('RGB')
    lr_pil, hr_t, lr_t = process_image(img, crop_size, low_res_size)

    # 6a) Your SR model
    with torch.no_grad():
        sr_t = model(lr_t)
    mse_sr = F.mse_loss(sr_t, hr_t).item()
    psnr_sr = -10 * np.log10(mse_sr)
    l1_sr = F.l1_loss(sr_t, hr_t).item()
    sr_np = (sr_t.cpu().squeeze(0).permute(1,2,0).numpy() * 0.5 + 0.5)
    sr_np = np.clip(sr_np, 0, 1)

    # 6b) Fine-tuned Real-ESRGAN
    lr_bgr = cv2.cvtColor(np.array(lr_pil), cv2.COLOR_RGB2BGR)
    sr_ft_bgr, _ = upsampler_ft.enhance(lr_bgr, outscale=scale_factor)
    sr_ft = cv2.cvtColor(sr_ft_bgr, cv2.COLOR_BGR2RGB)/255.0
    sr_ft_t = torch.from_numpy(sr_ft.transpose(2,0,1)).unsqueeze(0).to(device)*2.0 - 1.0
    mse_ft = F.mse_loss(sr_ft_t, hr_t).item()
    psnr_ft = -10 * np.log10(mse_ft)
    l1_ft = F.l1_loss(sr_ft_t, hr_t).item()

    # 6c) Anime Real-ESRGAN
    sr_an_bgr, _ = upsampler_an.enhance(lr_bgr, outscale=scale_factor)
    sr_an = cv2.cvtColor(sr_an_bgr, cv2.COLOR_BGR2RGB)/255.0
    sr_an_t = torch.from_numpy(sr_an.transpose(2,0,1)).unsqueeze(0).to(device)*2.0 - 1.0
    mse_an = F.mse_loss(sr_an_t, hr_t).item()
    psnr_an = -10 * np.log10(mse_an)
    l1_an = F.l1_loss(sr_an_t, hr_t).item()

    # 6d) Original HR
    hr_np = (hr_t.cpu().squeeze(0).permute(1,2,0).numpy() * 0.5 + 0.5)
    hr_np = np.clip(hr_np, 0, 1)

    # Plot SR epoch
    ax = fig.add_subplot(gs[r, 0])
    ax.imshow(sr_np)
    ax.axis('off')
    ax.text(0.02, 0.02, f"{psnr_sr:.1f} dB\nL1: {l1_sr:.3f}", transform=ax.transAxes,
            ha='left', va='bottom', fontsize=18,
            color='white', bbox=dict(facecolor='black', alpha=0.6, pad=4))

    # Plot fine-tuned
    ax = fig.add_subplot(gs[r, 1])
    ax.imshow(sr_ft)
    ax.axis('off')
    ax.text(0.02, 0.02, f"{psnr_ft:.1f} dB\nL1: {l1_ft:.3f}", transform=ax.transAxes,
            ha='left', va='bottom', fontsize=18,
            color='white', bbox=dict(facecolor='black', alpha=0.6, pad=4))

    # Plot anime
    ax = fig.add_subplot(gs[r, 2])
    ax.imshow(sr_an)
    ax.axis('off')
    ax.text(0.02, 0.02, f"{psnr_an:.1f} dB\nL1: {l1_an:.3f}", transform=ax.transAxes,
            ha='left', va='bottom', fontsize=18,
            color='white', bbox=dict(facecolor='black', alpha=0.6, pad=4))

    # Plot original
    ax = fig.add_subplot(gs[r, -1])
    ax.imshow(hr_np)
    ax.axis('off')

# Final layout & save
plt.subplots_adjust(top=0.92, left=0.1, right=0.98, bottom=0.05)
plt.savefig(f"comparison_epoch_{epoch_to_compare}_realesrgan.png", dpi=300, bbox_inches="tight")
plt.show()
