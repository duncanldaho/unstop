import os
import tensorflow as tf
import tensorflow_hub as hub


# File extension sanity check.
def extension_check(user_input):
    """Checks if input file exists."""
    if os.path.isfile(user_input):
        return user_input
    else:
        print("Invalid input.")
        user_input = extension_check(input("Image path: "))
        return user_input


# Preprocess image data.
def preprocess(image):
    """Decode JPEG, cast to float32, normalize to [0, 1], add batch dim."""
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image


# Resize image data.
def resize(image):
    """Resize image to 256x256."""
    image = tf.image.resize(
        image, (256, 256), preserve_aspect_ratio=True, antialias=False
    )
    return image


# Postprocess image data.
def postprocess(image):
    """Remove batch dim., convert to RGB, cast to uint8, encode JPEG data."""
    image = tf.squeeze(image, axis=None) * 255.0
    image = tf.cast(image, tf.uint8)
    image = tf.io.encode_jpeg(image, quality=95)
    return image


# User input for image paths.
content_path = extension_check(input("Content image path: "))
style_path = extension_check(input("Style image path: "))
stylized_path = input("Write path: ")

# Load raw JPEG data.
content = tf.io.read_file(content_path)
style = tf.io.read_file(style_path)

# Load image stylization module.
hub_module = hub.load(
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
)

# Processes image data, generate and save stylized image.
outputs = hub_module(preprocess(content), resize(preprocess(style)))
tf.io.write_file(stylized_path, postprocess(outputs))
