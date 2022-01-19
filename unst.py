import tensorflow as tf
import tensorflow_hub as hub


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
content = tf.io.read_file("data/content.jpeg")
style = tf.io.read_file("data/style.jpeg")

# Load image stylization module.
hub_module = hub.load(
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
)

# Generate and save stylized image.
outputs = hub_module(preprocess(content), resize(preprocess(style)))
tf.io.write_file("data/stylized.jpeg", postprocess(outputs))
