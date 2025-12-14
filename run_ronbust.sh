#!/bin/bash
base_dir="./robust_out"
mkdir -p ${base_dir}

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    test_robust_maskclip.py \
    --checkpoint_path "output_dir_bs16/MaskCLIP_sd15_20250730_00_41_24/checkpoint-33.pth" \
    --test_data_json "my_test_data.json" \
    --model_setting_name "ViTL" \
    --image_size 512 \
    --test_batch_size 12 \
    --world_size 4 \
    --if_resizing \
    --edge_mask_width 7 \
    --num_workers 12 \
    --output_dir "$base_dir" \
    --log_dir "$base_dir" \
    1> ${base_dir}/logs.log \
    2> ${base_dir}/error.log