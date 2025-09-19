from skimage import morphology
import tensorflow as tf
import numpy as np
import random
from matplotlib.pyplot import plt
import os



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

img = images[0].copy()
img_norm = img / np.max(img)

# Extract RGB
rgb = img_norm[:, :, :3]
R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

# Compute brightness
brightness = (R + G + B)/3

# Step 1: Find the brightest pixel
max_pos = np.unravel_index(np.argmax(brightness), brightness.shape)
print("Brightest pixel at:", max_pos)

# Step 2: Create initial mask with the brightest pixel
mask = np.zeros_like(brightness, dtype=bool)
mask[max_pos] = True

# Step 3: Region growing until brightness < threshold
threshold = 0.3  # minimum brightness to include (adjustable)
prev_mask = np.zeros_like(mask)
while not np.array_equal(prev_mask, mask):
    prev_mask = mask.copy()
    # Dilate current mask
    dilated = morphology.dilation(mask, morphology.square(3))
    # Add pixels that are bright enough
    mask = dilated & (brightness >= threshold)

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
img = images[i]
rgb_img = img[:, :, :3]  # Take only first 3 channels (RGB)
rgb_normalized = rgb_img / rgb_img.max()

ax[0].imshow(rgb)
ax[0].set_title("True Color (RGB)")
ax[0].axis("off")

ax[1].imshow(masks[i])
ax[1].set_title("Mask (Ground Truth)")
ax[1].axis("off")

ax[2].imshow(mask)
ax[2].set_title("Flood filling")
ax[2].axis("off")

plt.show()
