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

# -------------------------------
# 1. Prepare data (from your lists)
# -------------------------------
images = np.array(images)   # shape: (N, 512, 512, 4)  # 4 if RGB+NIR
masks = np.array(masks)     # shape: (N, 512, 512, 1)

# Normalize images to [0,1]
images = images.astype("float32") / 255.0
masks = (masks > 0).astype("float32")  # binary mask

# -------------------------------
# 2. Data augmentation pipeline
# -------------------------------
def augment(img, mask):
    # Ensure mask has channel dimension
    if tf.rank(mask) == 2:
        mask = tf.expand_dims(mask, axis=-1)

    # Random flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    # Random rotation
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)

    return img, mask

# -------------------------------
# 2. Preprocessing
# -------------------------------
def preprocess(img, mask):
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)

    # Ensure mask always has channel dimension
    mask = tf.expand_dims(mask, axis=-1) if mask.shape.rank == 2 else mask

    # Explicitly set shapes for TF graph
    img.set_shape([512, 512, img.shape[-1]])   # 3 or 4 channels depending on your data
    mask.set_shape([512, 512, 1])

    return img, mask

# -------------------------------
# 3. Dataset pipeline
# -------------------------------
BATCH_SIZE = 4
dataset = (
    tf.data.Dataset.from_tensor_slices((images, masks))
    .shuffle(100)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=True)   # ensures fixed batch size
    .prefetch(tf.data.AUTOTUNE)
)



# -------------------------------
# 4. Attention U-Net model
# -------------------------------
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def attention_block(x, g, filters):
    """Attention gate with proper shape alignment."""
    # Downsample x to match g's spatial dims
    theta_x = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(x)
    phi_g   = tf.keras.layers.Conv2D(filters, 1, padding="same")(g)

    add = tf.keras.layers.Add()([theta_x, phi_g])
    act = tf.keras.layers.ReLU()(add)
    psi = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(act)

    # Upsample attention coefficients back to x's original size
    psi = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(psi)

    # Multiply with skip connection
    return tf.keras.layers.Multiply()([x, psi])


def attention_unet(input_shape=(512,512,4)):
    inputs = tf.keras.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 64); p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1, 128); p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2, 256); p3 = tf.keras.layers.MaxPool2D()(c3)
    c4 = conv_block(p3, 512); p4 = tf.keras.layers.MaxPool2D()(c4)

    # Bottleneck
    c5 = conv_block(p4, 1024)

    # Decoder with attention
    g4 = tf.keras.layers.Conv2D(512, 1)(c5)
    a4 = attention_block(c4, g4, 512)
    u6 = tf.keras.layers.UpSampling2D()(c5)
    u6 = tf.keras.layers.Concatenate()([u6, a4])
    c6 = conv_block(u6, 512)

    g3 = tf.keras.layers.Conv2D(256, 1)(c6)
    a3 = attention_block(c3, g3, 256)
    u7 = tf.keras.layers.UpSampling2D()(c6)
    u7 = tf.keras.layers.Concatenate()([u7, a3])
    c7 = conv_block(u7, 256)

    g2 = tf.keras.layers.Conv2D(128, 1)(c7)
    a2 = attention_block(c2, g2, 128)
    u8 = tf.keras.layers.UpSampling2D()(c7)
    u8 = tf.keras.layers.Concatenate()([u8, a2])
    c8 = conv_block(u8, 128)

    g1 = tf.keras.layers.Conv2D(64, 1)(c8)
    a1 = attention_block(c1, g1, 64)
    u9 = tf.keras.layers.UpSampling2D()(c8)
    u9 = tf.keras.layers.Concatenate()([u9, a1])
    c9 = conv_block(u9, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c9)
    return tf.keras.Model(inputs, outputs)

model = attention_unet()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)])

# -------------------------------
# 5. Training
# -------------------------------
history = model.fit(dataset, epochs=30)


# -------------------------------
# 6. Inference on test images
# -------------------------------
preds = model.predict(dataset)  # predict on first 5 images
preds = (preds > 0.5).astype("uint8")  # threshold



for i in range(5):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # Input image (first 3 channels if RGB+NIR)
    img_show = images[i][..., :3] if images[i].shape[-1] > 3 else images[i]
    ax[0].imshow(img_show.astype(np.uint8))
    ax[0].set_title("Input Image")
    ax[0].axis("off")
    
    # Ground truth mask
    ax[1].imshow(masks[i].squeeze(), cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")
    
    # Predicted mask
    ax[2].imshow(preds[i].squeeze(), cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")
    
    plt.show()