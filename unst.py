import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# Define image loading and visualization functions
def show_n(images, titles=("",)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect="equal")
        plt.axis("off")
        plt.title(titles[i] if len(titles) > i else "")
    plt.savefig("data/stylized.jpeg")


# Load content and style images.
content_image = plt.imread("data/content.jpeg")
style_image = plt.imread("data/style.jpeg")

# Convert to float32 np array, add batch dimension, normalize to range [0, 1].
content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.0
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.0
# Optionally resize the images. It is recommended that the style image is about
# 256 pixels (this size was used when training the style transfer network).
# The content image can be any size.
style_image = tf.image.resize(style_image, (256, 256))

# Load image stylization module.
hub_module = hub.load(
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
)

# Stylize image:
# content_image, style_image, and stylized_image are expected to be 4-D Tensors
# with shapes [batch_size, image_height, image_width, 3].
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# Visualize input images and the generated stylized image.
show_n(
    [content_image, style_image, stylized_image],
    titles=["Original content image", "Style image", "Stylized image"],
)
