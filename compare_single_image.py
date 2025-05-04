import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import os

# Set style and increase font sizes for better readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Epoch checkpoints to compare
epochs = [50, 100, 200, 300, 450]

# Image path - replace with your actual image path
image_path = "dataset/assault_lily_bouquet_1.png"  # Update with your image path

# Function to process image
def process_image(img, crop_size=512, low_res_size=128):
    # Get dimensions
    width, height = img.size
    
    # If image is smaller than crop_size, resize it up
    if width < crop_size or height < crop_size:
        scale = crop_size / min(width, height) * 1.2  # Scale up a bit
        new_width, new_height = int(width * scale), int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        width, height = img.size
    
    # Calculate center crop coordinates
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    # Crop to square
    img_cropped = img.crop((left, top, right, bottom))
    
    # Create low-res version by resizing down
    img_low_res = img_cropped.resize((low_res_size, low_res_size), Image.BICUBIC)
    
    # Convert both to tensors [-1, 1] range as expected by your model
    to_tensor = T.ToTensor()
    hr_tensor = to_tensor(img_cropped) * 2.0 - 1.0
    lr_tensor = to_tensor(img_low_res) * 2.0 - 1.0
    
    return img_cropped, img_low_res, hr_tensor, lr_tensor

# Load your model
from SR import SRModel
model = SRModel().to(device)

# Load the image
original_img = Image.open(image_path).convert('RGB')
img_name = os.path.basename(image_path)

# Process the image
img_hr, img_lr, hr_tensor, lr_tensor = process_image(original_img)

# Add batch dimension
lr_tensor = lr_tensor.unsqueeze(0).to(device)
hr_tensor = hr_tensor.unsqueeze(0).to(device)

# Upsample low-res for display (traditional upscaling)
lr_upsampled = F.interpolate(lr_tensor, size=(512, 512), mode='bilinear', align_corners=False)
lr_np = lr_upsampled.cpu().squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5
lr_np = np.clip(lr_np, 0, 1)

# Original high-res
hr_np = hr_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5
hr_np = np.clip(hr_np, 0, 1)

# Create figure with more space at the top for column headers
fig, axes = plt.subplots(3, len(epochs), figsize=(len(epochs) * 4, 12))
fig.suptitle(f"Super-Resolution Model Comparison Across Epochs\nImage: {img_name}", fontsize=20, y=0.95)

# Add column headers with epoch numbers at the top of the figure
for i, epoch in enumerate(epochs):
    fig.text(0.15 + i * 0.175, 0.91, f"Epoch {epoch}", ha='center', fontsize=16)

# First row: Show low-res upscaled for all columns (no titles)
for i in range(len(epochs)):
    axes[0, i].imshow(lr_np)
    axes[0, i].axis('off')

# Process with each epoch model for the middle row
for i, epoch in enumerate(epochs):
    # Load model weights for this epoch
    model_path = f"weights/SR4_epoch_{epoch}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Run inference
        with torch.no_grad():
            # Get SR output
            sr_tensor = model(lr_tensor)
            
            # Convert to numpy for plotting
            sr_np = sr_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5
            sr_np = np.clip(sr_np, 0, 1)
        
        # Show SR result from this epoch in the middle row (no title)
        axes[1, i].imshow(sr_np)
        axes[1, i].axis('off')
        
        # Calculate PSNR (but don't display it as xlabel)
        psnr = -10 * torch.log10(F.mse_loss(sr_tensor, hr_tensor)).item()
        # Store PSNR value if needed later
    
    except Exception as e:
        print(f"Error processing epoch {epoch}: {e}")
        axes[1, i].text(0.5, 0.5, f"Error loading epoch {epoch}", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[1, i].transAxes)
        axes[1, i].axis('off')

# Third row: Show original high-res for all columns (no titles)
for i in range(len(epochs)):
    axes[2, i].imshow(hr_np)
    axes[2, i].axis('off')

# Add row labels
fig.text(0.01, 0.7, "Low-res Input", fontsize=16, rotation=90, va='center')
fig.text(0.01, 0.5, "SR Output", fontsize=16, rotation=90, va='center')
fig.text(0.01, 0.3, "Original High-res", fontsize=16, rotation=90, va='center')

plt.tight_layout(rect=[0.03, 0, 1, 0.9])  # Adjust to make room for the title and column headers
plt.savefig('epoch_comparison_single_image.png', dpi=300, bbox_inches='tight')
plt.show()