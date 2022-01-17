import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load content and style images.
content_image = plt.imread("data/content.jpeg")
style_image = plt.imread("data/style.jpeg")

# Convert to float32 np array, add batch dimension, normalize to range [0, 1].
content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.0
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.0

# It is recommended that the style image is 256 pixels,
# this size was used when training the style transfer network.
# style_image = tf.image.resize(style_image, (256, 256))
style_image = tf.image.resize(style_image, (256, 256), preserve_aspect_ratio=True, antialias=True)
# The content image can be any size.
# Load image stylization module.
hub_module = hub.load(
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
)

# Stylize image:
# content_image, style_image, and stylized_image are expected to be 4-D Tensors
# with shapes [batch_size, image_height, image_width, 3].
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# Convert stylized_iamge tensor to numpy array.
stylized_image_array = stylized_image[0].numpy()

# Save numpy array as image.
plt.imsave("data/stylized.jpeg", stylized_image_array)
