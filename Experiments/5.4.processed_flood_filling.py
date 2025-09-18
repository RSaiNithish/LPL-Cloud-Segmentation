from skimage import morphology
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

img = images[0].copy()
img_norm = img / np.max(img)

def preprocess(img, gamma=1.2):
    img = img.astype(float)
    img /= np.max(img)
    rgb = img[:,:,:3]
    nir = img[:,:,3]
    
    # Histogram stretching
    def stretch(c):
        p2, p98 = np.percentile(c, (2,98))
        c_stretched = (c - p2) / (p98 - p2)
        return np.clip(c_stretched, 0, 1)
    
    R, G, B = stretch(rgb[:,:,0]), stretch(rgb[:,:,1]), stretch(rgb[:,:,2])
    rgb_stretched = np.stack([R,G,B], axis=2) ** (1/gamma)
    
    brightness = np.mean(rgb_stretched, axis=2)
    ndvi = (nir - R) / (nir + R + 1e-6)
    
    return rgb_stretched, nir, brightness, ndvi

def region_grow(brightness, ndvi=None, min_thresh=0.3):
    max_pos = np.unravel_index(np.argmax(brightness), brightness.shape)
    mask = np.zeros_like(brightness, dtype=bool)
    mask[max_pos] = True
    prev_mask = np.zeros_like(mask)
    
    while not np.array_equal(prev_mask, mask):
        prev_mask = mask.copy()
        dilated = morphology.dilation(mask, morphology.square(3))
        if ndvi is not None:
            mask = dilated & (brightness >= min_thresh) & (ndvi < np.percentile(ndvi,40))
        else:
            mask = dilated & (brightness >= min_thresh)
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def detect_cloud_layers(img):
    rgb, nir, brightness, ndvi = preprocess(img)
    
    # White clouds (brightest)
    white_clouds = region_grow(brightness, ndvi, min_thresh=np.percentile(brightness,80))
    
    # Gray clouds (medium brightness)
    gray_mask = (brightness > np.percentile(brightness,70)) & (brightness <= np.percentile(brightness,90)) & (ndvi < np.percentile(ndvi,40))
    kernel = np.ones((3,3), np.uint8)
    gray_clouds = cv2.morphologyEx(gray_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    gray_clouds = cv2.morphologyEx(gray_clouds, cv2.MORPH_OPEN, kernel)
    
    return rgb, white_clouds, gray_clouds



img = images[3]
rgb_vis, white_mask, gray_mask = detect_cloud_layers(img)
fig, ax = plt.subplots(1, 3, figsize=(12, 6))


ax[0].imshow(rgb_vis)
ax[0].set_title("Enhanced (RGB)")
ax[0].axis("off")

ax[1].imshow(masks[3])
ax[1].set_title("Mask (Ground Truth)")
ax[1].axis("off")

ax[2].imshow(white_mask)
ax[2].set_title("Flood filling")
ax[2].axis("off")

plt.show()