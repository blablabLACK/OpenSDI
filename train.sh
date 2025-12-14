#!/bin/bash

base_dir="./debug"
mkdir -p ${base_dir}

# train with SD1.5 needs 4 GPUs
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./train.py \
    --exp_name MaskCLIP_sd15 \
    --model_setting_name 'ViTL' \
    --model MaskCLIP \
    --world_size 4 \
    --batch_size 16 \
    --data_path "balanced_dataset.json" \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 0 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --if_predict_label \
    --if_not_amp \
    --test_data_path "my_val_datasets.json" \
    --warmup_epochs 0 \
    --output_dir $base_dir \
    --log_dir $base_dir \
    --accum_iter 1 \
    --seed 42 \
    --test_period 1



