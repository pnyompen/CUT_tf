import os
import tensorflow as tf


def create_dir(dir):
    """ Create the directory.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} createrd')
    else:
        print(f'Directory {dir} already exists')

    return dir


@tf.function
def load_image(image_file, crop_size, load_size, preprocess='none', data_augmentation=True):
    """ Load the image file.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, expand_animations=False, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    if data_augmentation:
        image = tf.image.random_flip_left_right(image)
    if 'scale_shortside' in preprocess:
        image = resize_image_keep_aspect(image, load_size)
    if 'crop' in preprocess:
        image = tf.image.random_crop(image, (crop_size, crop_size, 3))
    if tf.shape(image)[-1] == 1:
        image = tf.tile(image, [1, 1, 3])

    return image


def resize_image_keep_aspect(image, lo_dim):
    # Take width/height
    initial_width = tf.cast(tf.shape(image)[0], tf.float32)
    initial_height = tf.cast(tf.shape(image)[1], tf.float32)

    # Take the greater value, and use it for the ratio
    min_ = tf.minimum(initial_width, initial_height)
    ratio = min_ / tf.constant(lo_dim, dtype=tf.float32)

    new_width = tf.cast(initial_width / ratio, tf.int32)
    new_height = tf.cast(initial_height / ratio, tf.int32)

    return tf.image.resize(image, [new_width, new_height])
