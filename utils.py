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
    image = tf.image.decode_png(image)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    if data_augmentation:
        image = tf.image.random_flip_left_right(image)
    if 'scale_shortside' in preprocess:
        c, w, h = image.shape
        if w > h:
            w = tf.cast(load_size / h * w, tf.int)
            h = load_size
        else:
            h = tf.cast(load_size / w * h, tf.int)
            w = load_size
        image = tf.image.resize(image, size=(w, h))
    if 'crop' in preprocess:
        image = tf.image.random_crop(image, (crop_size, crop_size))
    if tf.shape(image)[-1] == 1:
        image = tf.tile(image, [1, 1, 3])

    return image
