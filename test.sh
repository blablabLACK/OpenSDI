#!/bin/bash

base_dir="./debug"
mkdir -p ${base_dir}

torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./test.py \
    --model MaskCLIP \
    --model_setting_name 'ViTL' \
    --edge_mask_width 7 \
    --world_size 1 \
    --checkpoint_path "output_dir_bs16/MaskCLIP_sd15_20250730_00_41_24/checkpoint-33.pth" \
    --test_data_json "open_sdi_test_datasets.json" \
    --test_batch_size 1 \
    --image_size 512 \
    --if_resizing \
    --output_dir "$base_dir" \
    --log_dir "$base_dir"
