import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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

def cloud_mask_rgb_nir(img, gamma=1.2, bright_thresh=0.7, diff_thresh=0.1, nir_thresh=0.5, ndvi_thresh=0.2):
    """
    Cloud mask using RGB + NIR:
    - Brightness (close to white in RGB)
    - Low color difference (neutral)
    - High NIR reflectance
    - Low NDVI (exclude vegetation)
    """
    img = img.astype(float)
    img /= np.max(img)

    # Extract channels
    R, G, B, NIR = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]
    
    # Contrast stretching (same as before)
    def stretch(c):
        p2, p98 = np.percentile(c, (2, 98))
        c_stretched = (c - p2) / (p98 - p2)
        return np.clip(c_stretched, 0, 1)

    R, G, B, NIR = stretch(R), stretch(G), stretch(B), stretch(NIR)
    rgb_stretched = np.stack([R,G,B], axis=2) ** (1/gamma)
    
    # Brightness
    brightness = np.mean(rgb_stretched, axis=2)
    
    # Color neutrality
    diff = np.max(rgb_stretched, axis=2) - np.min(rgb_stretched, axis=2)
    
    # NDVI (vegetation index)
    ndvi = (NIR - R) / (NIR + R + 1e-6)
    
    # Cloud condition:
    mask = (
        (brightness > bright_thresh) &   # bright
        (diff < diff_thresh) &           # neutral
        (NIR > nir_thresh) &             # strong NIR reflection
        (ndvi < ndvi_thresh)             # not vegetation
    )
    
    mask_vis = (mask.astype(np.uint8)) * 255
    
    return rgb_stretched, mask_vis, brightness, diff, NIR, ndvi


# Example usage:
# img = images[2]   # should have shape (H,W,4) with R,G,B,NIR
rgb_stretched, mask_vis, brightness, diff, NIR, ndvi = cloud_mask_rgb_nir(images[9])

# Show results
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
plt.title("Enhanced RGB")
plt.imshow(rgb_stretched)
plt.axis("off")

plt.subplot(2,3,2)
plt.title("Cloud Mask (Ground Truth)")
plt.imshow(masks[9], cmap="gray")
plt.axis("off")

plt.subplot(2,3,3)
plt.title("Cloud Mask (RGB+NIR)")
plt.imshow(mask_vis, cmap="gray")
plt.axis("off")

plt.subplot(2,3,4)
plt.title("Difference (Neutrality)")
plt.imshow(diff, cmap="gray")
plt.axis("off")

plt.subplot(2,3,5)
plt.title("NIR Band")
plt.imshow(NIR, cmap="gray")
plt.axis("off")

plt.subplot(2,3,6)
plt.title("NDVI")
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(shrink=0.6)
plt.axis("off")

plt.show()
