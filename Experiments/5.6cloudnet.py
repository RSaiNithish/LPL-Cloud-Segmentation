import tensorflow as tf
import numpy as np
import random
from matplotlib.pyplot import plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
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


# ---------------------------
# Load pretrained Cloud-Net
# ---------------------------
model = load_model("Cloud-Net_trained_on_38-Cloud_training_patches.h5", compile=False)

# ---------------------------
# Function: Preprocess image (from array)
# ---------------------------
def preprocess_image_array(img):
    """
    Preprocess a single 4-band (RGB+NIR) image of size 512x512 (from np array).
    """
    img = img.astype("float32")
    
    # Normalize to [0,1]
    img /= np.max(img)
    
    # Resize to 192x192 (Cloud-Net input size)
    img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)
    
    # Add batch dimension
    img_input = np.expand_dims(img_resized, axis=0)  # (1,192,192,4)
    
    return img, img_input

# ---------------------------
# Function: Predict mask
# ---------------------------
def predict_mask_from_array(img):
    original_img, img_input = preprocess_image_array(img)
    
    # Predict
    pred = model.predict(img_input)
    mask_resized = cv2.resize(pred[0,:,:,0], (512, 512))  # resize back to 512x512
    
    # Threshold to get binary mask
    binary_mask = (mask_resized > 0.5).astype(np.uint8)
    
    return original_img, mask_resized, binary_mask


# ---------------------------
# Run test on first image in `images`
# ---------------------------
i=1
original_img, prob_mask, bin_mask = predict_mask_from_array(images[2])

# ---------------------------
# Visualize
# ---------------------------
fig, axs = plt.subplots(1,3, figsize=(15,5))
axs[0].imshow(original_img[:,:,:3])  # RGB only for visualization
axs[0].set_title("Original RGB")
axs[0].axis(False)
axs[1].imshow(prob_mask, cmap="gray")
axs[1].set_title("Predicted Cloud Probability")
axs[1].axis(False)
axs[2].imshow(masks[i], cmap="gray")
axs[2].set_title("Cloud Mask [Ground Truth]")
axs[2].axis(False)

plt.show()