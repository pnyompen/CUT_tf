""" USAGE
python ./train.py --train_src_dir ./datasets/horse2zebra/trainA --train_tar_dir ./datasets/horse2zebra/trainB --test_src_dir ./datasets/horse2zebra/testA --test_tar_dir ./datasets/horse2zebra/testB

python scripts/save_model.py --out_path='/Volumes/T5/Workspace/neural_camera/assets/cut.tflite' --ckpt ckpts/resnet
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
from PIL import Image

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
    parser.add_argument('--tmp_dir', help='Outputs folder',
                        type=str, default='./output')
    parser.add_argument('--out_path', help='Outputs folder',
                        type=str, default='/Volumes/T5/Workspace/neural_camera/neural_camera/assets/test.tflite')
    # Misc
    parser.add_argument(
        '--ckpt', help='Resume training from checkpoint', type=str)

    args = parser.parse_args()

    # Check arguments
    return args


def main(args):
    source_shape = target_shape = [384, 216, 3]
    # source_shape = target_shape = [336, 192, 3]

    # Create model
    cut = CUT_model(source_shape, target_shape,
                    cut_mode=args.mode, impl=args.impl,
                    norm_layer='instance', ngf=16, ndf=32,
                    use_antialias=True,
                    resnet_blocks=4,
                    downsample_blocks=2,
                    netF_units=256,
                    netF_num_patches=256,
                    nce_layers=[0, 3, 4, 5, 6])
    cut.summary()
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

    # Define the folders to store output information
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_path)

    test_img_path = 'test/test.jpg'
    img = Image.open(test_img_path).resize(source_shape[:2][::-1])
    img = (np.array(img) / 127.5) - 1.0

    # add N dim
    input_data = np.expand_dims(img, axis=0)
    synthesized = model.predict(input_data)[0]
    synthesized = (synthesized + 1) * 127.5
    synthesized = np.array(synthesized, dtype=np.uint8)
    im = Image.fromarray(synthesized)
    im.save(tmp_dir / 'temp.png')

    # モデルを保存
    export_dir = str(tmp_dir / 'tmp_model')
    tf.saved_model.save(model, export_dir)

    # モデルを変換
    # converter = tf.lite.TFLiteConverter.from_saved_model(
    #     export_dir,
    # )
    model = tf.saved_model.load(export_dir)
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, *source_shape])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # quantize
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    shutil.rmtree(export_dir)

    print('success')

if __name__ == '__main__':
    main(ArgParse())
