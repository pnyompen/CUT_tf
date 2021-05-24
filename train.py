""" USAGE
python ./train.py --train_src_dir ./datasets/horse2zebra/trainA --train_tar_dir ./datasets/horse2zebra/trainB --test_src_dir ./datasets/horse2zebra/testA --test_tar_dir ./datasets/horse2zebra/testB
"""

import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from adabelief_tf import AdaBeliefOptimizer

from modules.cut_model import CUT_model
from utils import create_dir, load_image


def ArgParse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CUT training usage.')
    # Training
    parser.add_argument('--mode', help="Model's mode be one of: 'cut', 'fastcut'",
                        type=str, default='cut', choices=['cut', 'fastcut'])
    parser.add_argument('--gan_mode', help='The type of GAN objective.',
                        type=str, default='lsgan', choices=['lsgan', 'nonsaturating', 'hinge'])
    parser.add_argument(
        '--epochs', help='Number of training epochs', type=int, default=400)
    parser.add_argument(
        '--batch_size', help='Training batch size', type=int, default=1)
    parser.add_argument(
        '--beta_1', help='First Momentum term of adam', type=float, default=0.5)
    parser.add_argument(
        '--beta_2', help='Second Momentum term of adam', type=float, default=0.999)
    parser.add_argument(
        '--lr', help='Initial learning rate for adam', type=float, default=0.0002)
    parser.add_argument('--lr_decay_rate',
                        help='lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--lr_decay_step',
                        help='lr_decay_step', type=int, default=100000)
    parser.add_argument('--vgg_lambda', type=int, default=10,
                        help='Weight about perceptual loss')
    # Define data
    parser.add_argument('--out_dir', help='Outputs folder',
                        type=str, default='./output')
    parser.add_argument('--train_src_dir', help='Train-source dataset folder',
                        type=str, default='./datasets/horse2zebra/trainA')
    parser.add_argument('--train_tar_dir', help='Train-target dataset folder',
                        type=str, default='./datasets/horse2zebra/trainB')
    parser.add_argument('--train_tar_pattern', help='Train-target dataset folder',
                        type=str, default='*')
    parser.add_argument('--test_src_dir', help='Test-source dataset folder',
                        type=str, default='./datasets/horse2zebra/testA')
    parser.add_argument('--test_tar_dir', help='Test-target dataset folder',
                        type=str, default='./datasets/horse2zebra/testB')
    # Misc
    parser.add_argument(
        '--ckpt', help='Resume training from checkpoint', type=str)
    parser.add_argument(
        '--save_n_epoch', help='Every n epochs to save checkpoints', type=int, default=5)
    parser.add_argument('--impl', help="(Faster)Custom op use:'cuda'; (Slower)Tensorflow op use:'ref'",
                        type=str, default='ref', choices=['ref', 'cuda'])
    # Dataset
    parser.add_argument('--preprocess', type=str, default='none',
                        choices=['none', 'crop', 'scale_shortside_and_crop'])
    parser.add_argument('--crop_size', type=str, default="384,216")
    parser.add_argument('--load_size', type=int, default=384)
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--src_data_augmentation', action='store_true')
    parser.add_argument('--use_diffaugment', action='store_true')
    parser.add_argument('--use_antialias', action='store_true')
    parser.add_argument('--memory_growth', action='store_true')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--optimizer', type=str,
                        default='adam', help='(adam|adaBelief)')

    args = parser.parse_args()

    # Check arguments
    assert args.lr > 0
    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.save_n_epoch > 0
    assert os.path.exists(
        args.train_src_dir), 'Error: Train source dataset does not exist.'
    assert os.path.exists(
        args.train_tar_dir), 'Error: Train target dataset does not exist.'
    assert os.path.exists(
        args.test_src_dir), 'Error: Test source dataset does not exist.'
    assert os.path.exists(
        args.test_tar_dir), 'Error: Test target dataset does not exist.'

    return args


def main(args):

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0 and args.memory_growth:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(
                device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")
        # Create datasets
    train_dataset, test_dataset = create_dataset(args)

    # Get image shape
    source_image, target_image = next(iter(train_dataset))
    source_shape = source_image.shape[1:]
    target_shape = target_image.shape[1:]

    # Create model
    cut = CUT_model(source_shape, target_shape,
                    cut_mode=args.mode, impl=args.impl, norm_layer='instance',
                    use_antialias=args.use_antialias, ndf=args.ndf, ngf=args.ngf,
                    resnet_blocks=4,
                    downsample_blocks=2,
                    netF_units=256,
                    netF_num_patches=256,
                    nce_layers=[0, 3, 4, 5, 6],
                    use_diffaugment=args.use_diffaugment,
                    gan_mode=args.gan_mode,
                    vgg_lambda=args.vgg_lambda
                    )
    cut.summary()
    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr,
                                                                 decay_steps=args.lr_decay_step,
                                                                 decay_rate=args.lr_decay_rate,
                                                                 staircase=True)
    if args.optimizer == 'adam':
        G_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2)
        F_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2)
        D_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2)
    elif args.optimizer == 'adaBelief':
        G_optimizer = AdaBeliefOptimizer(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=1e-12, rectify=False)
        F_optimizer = AdaBeliefOptimizer(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=1e-12, rectify=False)
        D_optimizer = AdaBeliefOptimizer(
            learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=1e-12, rectify=False)
    else:
        raise Exception('Invalid optimizer')
    # Compile model
    cut.compile(G_optimizer=G_optimizer,
                F_optimizer=F_optimizer,
                D_optimizer=D_optimizer,)

    # Restored from previous checkpoints, or initialize checkpoints from scratch
    if args.ckpt is not None:
        latest_ckpt = tf.train.latest_checkpoint(args.ckpt)
        cut.load_weights(latest_ckpt)
        initial_epoch = int(latest_ckpt[-3:])
        print(f"Restored from {latest_ckpt}.")
    else:
        initial_epoch = 0
        print("Initializing from scratch...")

    # Define the folders to store output information
    result_dir = f'{args.out_dir}/images'
    checkpoint_dir = f'{args.out_dir}/checkpoints'
    log_dir = f'{args.out_dir}/logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    # Create validating callback to generate output image every epoch
    plotter_callback = GANMonitor(cut.netG, test_dataset, result_dir)

    # Create checkpoint callback to save model's checkpoints every n epoch (default 5)
    # Use period to save every n epochs, use save_freq to save every n batches
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+'/{epoch:03d}', period=args.save_n_epoch, verbose=1)
    # Create tensorboard callback to log losses every epoch
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Train cut model
    cut.fit(train_dataset,
            epochs=args.epochs,
            initial_epoch=initial_epoch,
            callbacks=[plotter_callback,
                       checkpoint_callback,
                    #    tensorboard_callback
                       ],
            workers=args.n_workers,
            steps_per_epoch=args.steps_per_epoch,
            verbose=1)


def create_dataset(args):
    """ Create tf.data.Dataset.
    """
    # Create train dataset
    crop_size = list(map(int, args.crop_size.split(',')))
    train_src_dataset = tf.data.Dataset.list_files(
        [args.train_src_dir+'/*.jpg', args.train_src_dir+'/*.jpeg', args.train_src_dir+'/*.png'], shuffle=True)
    train_src_dataset = (
        train_src_dataset.map(lambda x: load_image(x, crop_size=crop_size, load_size=args.load_size,
                                                   preprocess=args.preprocess, src_data_augmentation=args.src_data_augmentation), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(args.batch_size, drop_remainder=True).repeat()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    train_tar_dataset = tf.data.Dataset.list_files(
        [
            args.train_tar_dir+f'/{args.train_tar_pattern}.jpg',
            args.train_tar_dir+f'/{args.train_tar_pattern}.jpeg',
            args.train_tar_dir+f'/{args.train_tar_pattern}.png'], shuffle=True)
    train_tar_dataset = (
        train_tar_dataset.map(lambda x: load_image(x, crop_size=crop_size, load_size=args.load_size,
                                                   preprocess=args.preprocess), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(args.batch_size, drop_remainder=True).repeat()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    train_dataset = tf.data.Dataset.zip((train_src_dataset, train_tar_dataset))

    # Create test dataset
    test_src_dataset = tf.data.Dataset.list_files(
        [args.test_src_dir+'/*.jpg', args.test_src_dir+'/*.jpeg', args.test_src_dir+'/*.png'])
    test_src_dataset = (
        test_src_dataset.map(lambda x: load_image(x, crop_size=crop_size, load_size=args.load_size,
                                                  preprocess=args.preprocess, src_data_augmentation=args.src_data_augmentation), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(args.batch_size, drop_remainder=True).repeat()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_tar_dataset = tf.data.Dataset.list_files(
        [args.test_tar_dir+f'/{args.train_tar_pattern}.jpg',
         args.test_tar_dir+f'/{args.train_tar_pattern}.jpeg',
         args.test_tar_dir+f'/{args.train_tar_pattern}.png'])
    test_tar_dataset = (
        test_tar_dataset.map(lambda x: load_image(x, crop_size=crop_size, load_size=args.load_size,
                                                  preprocess=args.preprocess), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(args.batch_size, drop_remainder=True).repeat()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_dataset = tf.data.Dataset.zip((test_src_dataset, test_tar_dataset))

    return train_dataset, test_dataset


class GANMonitor(tf.keras.callbacks.Callback):
    """ A callback to generate and save images after each epoch
    """

    def __init__(self, generator, test_dataset, out_dir, num_img=2):
        self.num_img = num_img
        self.generator = generator
        self.test_dataset = test_dataset
        self.out_dir = create_dir(out_dir)

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(self.num_img, 4, figsize=(20, 10))
        [ax[0, i].set_title(title) for i, title in enumerate(
            ['Source', "Translated", "Target", "Identity"])]
        for i, (source, target) in enumerate(self.test_dataset.take(self.num_img)):
            translated = self.generator(source)[0].numpy()
            translated = (translated * 127.5 + 127.5).astype(np.uint8)
            source = (source[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            idt = self.generator(target)[0].numpy()
            idt = (idt * 127.5 + 127.5).astype(np.uint8)
            target = (target[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            [ax[i, j].imshow(img) for j, img in enumerate(
                [source, translated, target, idt])]
            [ax[i, j].axis("off") for j in range(4)]

        plt.savefig(f'{self.out_dir}/epoch={epoch + 1}.png')
        plt.close()


if __name__ == '__main__':
    main(ArgParse())
