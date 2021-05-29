import os
import tensorflow as tf
import random


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
def load_image(image_file, crop_size, load_size, preprocess='none', data_augmentation=True,
               src_data_augmentation=False, tar_data_augmentation=False):
    """ Load the image file.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, expand_animations=False, channels=3)

    if src_data_augmentation:
        image = tf.image.random_brightness(image, max_delta=0.25)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    if tar_data_augmentation:
        image = tf.keras.preprocessing.image.random_rotation(
            image, 25, fill_mode='reflect', row_axis=0, col_axis=1, channel_axis=2)
        zoom_ratio = random.random() * 1 + 0.75
        image = tf.keras.preprocessing.image.random_zoom(
            image, (zoom_ratio, zoom_ratio), fill_mode='reflect', row_axis=0, col_axis=1, channel_axis=2)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    if 'scale_shortside' in preprocess:
        image = resize_image_keep_aspect(image, load_size)
    if 'crop' in preprocess:
        image = tf.image.random_crop(image, (*crop_size, 3))
    if data_augmentation:
        image = tf.image.random_flip_left_right(image)
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
    new_width = new_height = lo_dim
    if initial_width > initial_height:
        new_width = tf.cast(initial_width / ratio, tf.int32)
    else:
        new_height = tf.cast(initial_height / ratio, tf.int32)

    return tf.image.resize(image, [new_width, new_height])
