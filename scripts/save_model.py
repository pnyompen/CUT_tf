""" USAGE
python ./train.py --train_src_dir ./datasets/horse2zebra/trainA --train_tar_dir ./datasets/horse2zebra/trainB --test_src_dir ./datasets/horse2zebra/testA --test_tar_dir ./datasets/horse2zebra/testB
"""

import sys
import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

sys.path.append('./')
from modules.cut_model import CUT_model
from utils import create_dir, load_image

tf.get_logger().setLevel('ERROR')

def ArgParse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CUT training usage.')
    # Training
    parser.add_argument('--mode', help="Model's mode be one of: 'cut', 'fastcut'",
                        type=str, default='cut', choices=['cut', 'fastcut'])
    parser.add_argument('--impl', help="(Faster)Custom op use:'cuda'; (Slower)Tensorflow op use:'ref'",
                        type=str, default='ref', choices=['ref', 'cuda'])
    parser.add_argument('--out_dir', help='Outputs folder',
                        type=str, default='./output')
    # Misc
    parser.add_argument(
        '--ckpt', help='Resume training from checkpoint', type=str)

    args = parser.parse_args()

    # Check arguments
    return args


def main(args):
    source_shape = target_shape = [256, 256, 3]

    # Create model
    cut = CUT_model(source_shape, target_shape,
                    cut_mode=args.mode, impl=args.impl, model='resnet', norm_layer='batch')
    # Restored from previous checkpoints, or initialize checkpoints from scratch
    if args.ckpt:
        latest_ckpt = tf.train.latest_checkpoint(args.ckpt)
        cut.load_weights(latest_ckpt)
        model = cut.netG
        # model.compile()
        print(f"Restored from {latest_ckpt}.")
    else:
        model = cut.netG
        model.compile()

    model.summary()

    # Define the folders to store output information
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # モデルを保存
    export_dir = str(out_dir / 'tmp_model')
    tf.saved_model.save(model, export_dir)

    # モデルを変換
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)


    # converter.allow_custom_ops = True
    # quantize
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    out_path = out_dir / 'cut.tflite'
    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    shutil.rmtree(export_dir)

    print('success')


if __name__ == '__main__':
    main(ArgParse())
