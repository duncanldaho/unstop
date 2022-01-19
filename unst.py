import os
import tensorflow as tf
import tensorflow_hub as hub


# JPEG/JPG compatibility.
def get_image_path(data_dir, image_name):
    """Checks data_dir for image_name with file ext. JPEG & JPG"""
    jpeg_path = os.path.join(data_dir, "{}.jpeg".format(image_name))
    if os.path.exists(jpeg_path):
        return jpeg_path

    jpg_path = os.path.join(data_dir, "{}.jpg".format(image_name))
    if os.path.exists(jpg_path):
        return jpg_path


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
        image, (256, 256), preserve_aspect_ratio=True, antialias=True
    )
    return image


# Postprocess image data.
def postprocess(image):
    """Remove batch dim., convert to RGB, cast to uint8, encode JPEG data."""
    image = tf.squeeze(image, axis=None) * 255.0
    image = tf.cast(image, tf.uint8)
    image = tf.io.encode_jpeg(image, quality=95)
    return image


# Load raw JPEG data.
content = tf.io.read_file(get_image_path("data/", "content"))
style = tf.io.read_file(get_image_path("data/", "style"))

# Load image stylization module.
hub_module = hub.load(
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
)

# Processes image data, generate and save stylized image.
outputs = hub_module(preprocess(content), resize(preprocess(style)))
tf.io.write_file("data/stylized.jpeg", postprocess(outputs))
