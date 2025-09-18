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




def normalize(arr):
    arr = arr.astype(float)
    arr_min, arr_max = arr.min(), arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

i=3
fc = normalize(images[i][:,:,3].copy())


t = 0.6
fc[fc<t]=0
fc[fc>=t] = 1

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
img = images[i]
rgb_img = img[:, :, :3]  # Take only first 3 channels (RGB)
rgb_normalized = rgb_img / rgb_img.max()

ax[0].imshow(rgb_normalized)
ax[0].set_title("True Color (RGB)")
ax[0].axis("off")

ax[1].imshow(masks[i])
ax[1].set_title("Mask (Ground Truth)")
ax[1].axis("off")

ax[2].imshow(fc)
ax[2].set_title("Manual Thresholding")
ax[2].axis("off")

plt.show()
