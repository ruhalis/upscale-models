import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Real-ESRGAN imports (make sure realesrgan is installed)
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ─── 1) Style & Globals ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20
})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = torch.cuda.is_available()  # half precision only on GPU

# ─── 2) Paths & settings ──────────────────────────────────────────────────────
image_paths = [
#    "C:/Users/ruhalis/Documents/upscale-models/dataset/assault_lily_bouquet_1.png",
#    "C:/Users/ruhalis/Documents/upscale-models/dataset/violet_evergarden_0.png",
    "C:/Users/ruhalis/Documents/upscale-models/dataset/magia_record_s1_0.png"
]
crop_size    = 512
low_res_size = 128
scale_factor = 4  # both models are x4

# ─── 3) Define your two Real-ESRGAN upsamplers ────────────────────────────────
# 3a) Fine-tuned Real-ESRGAN (replace path with your own weights)
finetuned_weights = "experiments/finetune_RealESRGANx4plus_400k/models/net_g_3000.pth"
model_ft = RRDBNet(num_in_ch=3, num_out_ch=3,
                   num_feat=64, num_block=23, num_grow_ch=32,
                   scale=scale_factor)
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

# 3b) Official “anime” Real-ESRGAN
anime_weights = "weights/RealESRGAN_x4plus_anime_6B.pth"
model_anime = RRDBNet(num_in_ch=3, num_out_ch=3,
                      num_feat=64, num_block=6, num_grow_ch=32,
                      scale=scale_factor)
upsampler_anime = RealESRGANer(
    scale=scale_factor,
    model_path=anime_weights,
    dni_weight=None,
    model=model_anime,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=0 if torch.cuda.is_available() else None
)

# ─── 4) Image preprocessing helper ────────────────────────────────────────────
def process_image(img: Image.Image,
                  crop_size: int = 512,
                  low_res_size: int = 128):
    """Center-crop to crop_size, downscale to low_res_size, return tensors."""
    w, h = img.size
    if min(w, h) < crop_size:
        scale = crop_size / min(w, h) * 1.2
        img   = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        w, h  = img.size

    left = (w - crop_size)//2
    top  = (h - crop_size)//2
    crop = img.crop((left, top, left+crop_size, top+crop_size))
    lr   = crop.resize((low_res_size, low_res_size), Image.BICUBIC)

    to_t = T.ToTensor()
    hr_t = to_t(crop)*2.0 - 1.0
    lr_t = to_t(lr)  *2.0 - 1.0
    return crop, lr, hr_t.unsqueeze(0), lr_t.unsqueeze(0)

# ─── 5) Build matplotlib grid ─────────────────────────────────────────────────
n_rows = len(image_paths)
n_cols = 1 + 2 + 1  # bicubic | fine-tuned | anime | original

fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
gs  = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.05)

col_labels = [
    "Bicubic\nUpscaled",
    "Fine-tuned\nReal-ESRGAN",
    "Real-ESRGAN\nAnime",
    "Original\nHigh-res"
]
for c, label in enumerate(col_labels):
    ax = fig.add_subplot(gs[0, c])
    ax.set_title(label, pad=8)
    ax.axis("off")

# ─── 6) Loop over images & compute metrics ────────────────────────────────────
for r, img_path in enumerate(image_paths):
    # load & preprocess
    img = Image.open(img_path).convert("RGB")
    _, lr_pil, hr_t, lr_t = process_image(img, crop_size, low_res_size)
    hr_t = hr_t.to(device)
    lr_t = lr_t.to(device)

    # 6a) Bicubic baseline
    with torch.no_grad():
        bicubic_t = F.interpolate(
            lr_t,
            size=(hr_t.shape[2], hr_t.shape[3]),
            mode="bilinear",
            align_corners=False
        )
    mse_b = F.mse_loss(bicubic_t, hr_t).item()
    psnr_b = -10 * np.log10(mse_b)
    l1_b   = F.l1_loss(bicubic_t, hr_t).item()

    bicubic_np = bicubic_t.cpu().squeeze(0).permute(1,2,0).numpy()*0.5 + 0.5
    bicubic_np = np.clip(bicubic_np,0,1)

    # 6b) Fine-tuned Real-ESRGAN
    lr_bgr = cv2.cvtColor(np.array(lr_pil), cv2.COLOR_RGB2BGR)
    sr_ft_bgr, _ = upsampler_ft.enhance(lr_bgr, outscale=scale_factor)
    sr_ft_rgb   = cv2.cvtColor(sr_ft_bgr, cv2.COLOR_BGR2RGB)/255.0
    sr_ft_t     = torch.from_numpy(sr_ft_rgb.transpose(2,0,1)).unsqueeze(0).to(device)*2.0 - 1.0
    mse_ft = F.mse_loss(sr_ft_t, hr_t).item()
    psnr_ft = -10 * np.log10(mse_ft)
    l1_ft   = F.l1_loss(sr_ft_t, hr_t).item()

    # 6c) Official Anime model
    sr_an_bgr, _ = upsampler_anime.enhance(lr_bgr, outscale=scale_factor)
    sr_an_rgb   = cv2.cvtColor(sr_an_bgr, cv2.COLOR_BGR2RGB)/255.0
    sr_an_t     = torch.from_numpy(sr_an_rgb.transpose(2,0,1)).unsqueeze(0).to(device)*2.0 - 1.0
    mse_an = F.mse_loss(sr_an_t, hr_t).item()
    psnr_an = -10 * np.log10(mse_an)
    l1_an   = F.l1_loss(sr_an_t, hr_t).item()

    # 6d) Original
    hr_np = hr_t.cpu().squeeze(0).permute(1,2,0).numpy()*0.5 + 0.5
    hr_np = np.clip(hr_np,0,1)

    # ── Plot Bicubic ───────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[r, 0])
    ax.imshow(bicubic_np)
    ax.axis("off")
    ax.text(
        0.98, 0.02,
        f"{psnr_b:.1f} dB\nL1: {l1_b:.3f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=18,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6, pad=4)
    )
    ax.set_ylabel(
        os.path.basename(img_path),
        rotation=0, labelpad=40, va="center", fontsize=14
    )

    # ── Plot Fine-tuned Real-ESRGAN ────────────────────────────────────────────
    ax = fig.add_subplot(gs[r, 1])
    ax.imshow(sr_ft_rgb)
    ax.axis("off")
    ax.text(
        0.02, 0.02,
        f"{psnr_ft:.1f} dB\nL1: {l1_ft:.3f}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=18,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6, pad=4)
    )

    # ── Plot Official Anime Real-ESRGAN ────────────────────────────────────────
    ax = fig.add_subplot(gs[r, 2])
    ax.imshow(sr_an_rgb)
    ax.axis("off")
    ax.text(
        0.02, 0.02,
        f"{psnr_an:.1f} dB\nL1: {l1_an:.3f}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=18,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6, pad=4)
    )

    # ── Plot Original High-res ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[r, -1])
    ax.imshow(hr_np)
    ax.axis("off")

# ─── 7) Final layout & save ───────────────────────────────────────────────────

plt.subplots_adjust(top=0.92, left=0.1, right=0.98, bottom=0.05)
plt.savefig("realesrgan_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
