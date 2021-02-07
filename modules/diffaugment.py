# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import tensorflow as tf
from easydict import EasyDict


def DiffAugment(x, policy='', params=None, channels_first=False):
    if params is None:
        params = get_params(x, policy)
    if policy:
        if channels_first:
            x = tf.transpose(x, [0, 2, 3, 1])
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x, params)
        if channels_first:
            x = tf.transpose(x, [0, 3, 1, 2])
    return x


def get_params(x, policy, translation_ratio=0.125, cutout_ratio=0.5):
    params = {}
    for p in policy.split(','):
        if p == 'color':
            params.update(
                brightness=tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5,
                saturation=tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2,
                contrast=tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
            )
        elif p == 'translation':
            image_size = tf.shape(x)[1:3]
            batch_size = tf.shape(x)[0]
            shift = tf.cast(tf.cast(image_size, tf.float32) *
                            translation_ratio + 0.5, tf.int32)
            params.update(
                translation_x=tf.random.uniform(
                    [batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32),
                translation_y=tf.random.uniform(
                    [batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32),
                translation_ratio=translation_ratio
            )
        elif p == 'cutout':
            cutout_size = tf.cast(tf.cast(image_size, tf.float32)
                                  * cutout_ratio + 0.5, tf.int32)
            params.update(
                cutout_offset_x=tf.random.uniform([tf.shape(
                    x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32),
                cutout_offset_y=tf.random.uniform([tf.shape(
                    x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32),
                cutout_size=cutout_size,
                cutout_ratio=cutout_ratio
            )

    return EasyDict(params)


def rand_brightness(x, params):
    x = x + params.brightness
    return x


def rand_saturation(x, params):
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * params.saturation + x_mean
    return x


def rand_contrast(x, params):
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * params.contrast + x_mean
    return x


def rand_translation(x, params):
    image_size = tf.shape(x)[1:3]
    translation_x = params.translation_x
    translation_y = params.translation_y
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(
        image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0, image_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(
        image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0, image_size[1] + 1)
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(
        grid_x, -1), batch_dims=1)
    x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [
                     1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return x


def rand_cutout(x, params):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = params.cutout_size
    offset_x = params.cutout_offset_x
    offset_y = params.cutout_offset_y
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(
        cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] //
                            2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(
        cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones(
        [batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
