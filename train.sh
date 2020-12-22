dataroot='datasets/test'
python train.py --mode cut                                    \
                --save_n_epoch 1                            \
                --train_src_dir "$dataroot/trainA" \
                --train_tar_dir "$dataroot/trainB" \
                --test_src_dir "$dataroot/trainA"   \
                --test_tar_dir "$dataroot/trainB"   \
                --load_size 320 --crop_size 256 \
                --preprocess scale_shortside_and_crop