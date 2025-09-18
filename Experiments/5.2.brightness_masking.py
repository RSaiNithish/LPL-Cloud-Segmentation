import tensorflow as tf
import numpy as np
import random
from matplotlib.pyplot import plt
import os
import cv2



path = 'images'
basenames = os.listdir(path)
images = []
masks = []
for b in basenames:
    if not b.endswith(".npy"):
        continue
    img_path = f'{path}/{b}' 
    img = np.load(img_path)
    print(img.shape)
    mask_path = img_path.replace("image","mask")
    print(os.path.exists(mask_path))
    mask = np.load(mask_path)
    print(mask.shape)
    images.append(img)
    masks.append(mask)


i =0
# Load RGB image (already in array 'images')
img = images[i]                          # shape (H, W, 3)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize image
rgb_norm = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Convert to grayscale (brightness proxy)
gray = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2GRAY)

# Pick percentile threshold (e.g., top 5% brightest pixels)
p = 80  # 95th percentile
thresh_val = np.percentile(gray, p)

# Create mask
_, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

# Apply mask to original image
white_only = cv2.bitwise_and(rgb_norm, rgb_norm, mask=mask)


# Show results
plt.subplot(1,3,1); plt.imshow(rgb_norm); plt.title("Normalized"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(masks[i],); plt.title("Mask (Ground Truth)"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(mask); plt.title("White Extracted"); plt.axis("off")
plt.show()
