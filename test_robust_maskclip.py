#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskCLIP Robustness Test (DDP-ready)
- Local-import IMDLBenCo-main
- RotationWrapper robustness sweep
- JsonDataset / ManiDataset auto
- Evaluator: PixelF1
"""

import os, sys, json, time, types, inspect, argparse, datetime
import numpy as np
from pathlib import Path

# ---------- Force local IMDLBenCo ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMDL_LOCAL = os.path.abspath(os.path.join(THIS_DIR, "../IMDLBenCo-main"))
print(f"[DEBUG] Local IMDLBenCo path: {IMDL_LOCAL}")
if IMDL_LOCAL not in sys.path:
    sys.path.insert(0, IMDL_LOCAL)

# ---------- Torch & TB ----------
import torch
from torch.utils.tensorboard import SummaryWriter

# ---------- IMDLBenCo imports (local) ----------
import IMDLBenCo.training_scripts.utils.misc as misc
from IMDLBenCo.registry import POSTFUNCS
from IMDLBenCo.datasets import ManiDataset, JsonDataset
from IMDLBenCo.training_scripts.tester import test_one_epoch
from IMDLBenCo.evaluation import PixelF1
from IMDLBenCo.transforms.robustness_wrapper import RotationWrapper,GaussianNoiseWrapper,ColorJitterWrapper,ResolutionChangeWrapper,JpegCompressionWrapper
# 注：如果你本地 robustness_wrapper 里还有其他 Wrapper，可在这里按需导入

# ---------- Your model ----------
from model.MaskCLIP import MaskCLIP


def get_args():
    parser = argparse.ArgumentParser("MaskCLIP Robustness Test", add_help=True)

    # --- model specific ---
    parser.add_argument("--model_setting_name", type=str, required=True,
                        help="e.g., ViTL / ViTB, must match MaskCLIP(main_keys)")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    # --- data / io ---
    parser.add_argument("--test_data_json", type=str, required=True,
                        help="JSON list like: [[name, path], ...]; path can be json file (JsonDataset) or folder (ManiDataset)")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--if_padding", action="store_true")
    parser.add_argument("--if_resizing", action="store_true")
    parser.add_argument("--edge_mask_width", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="./robust_out")
    parser.add_argument("--log_dir", type=str, default="./robust_out")

    # --- runtime ---
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # --- evaluators ---
    parser.add_argument("--threshold", type=float, default=0.5)

    # --- distributed (keep names compatible with misc.init_distributed_mode) ---
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", type=str, default="env://")

    # --- flags referenced inside tester ---
    parser.add_argument("--no_model_eval", action="store_true")

    args = parser.parse_args()
    return args


def build_model(args, device):
    print("[INFO] Loading MaskCLIP ...")
    model = MaskCLIP(model_setting_name=args.model_setting_name)
    model.to(device)
    return model


def load_ckpt_strict(model, ckpt_path, map_location="cuda"):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)


def build_loader(args, dataset_path, name, attack_transform):
    """
    dataset_path: can be a folder (ManiDataset) or a json file (JsonDataset)
    """
    # optional post function from registry (usually None for MaskCLIP)
    post_function_name = f"maskclip_post_func".lower()
    post_function = None
    try:
        if POSTFUNCS.has(post_function_name):
            post_function = POSTFUNCS.get_lower(post_function_name)
    except Exception:
        post_function = None

    if os.path.isdir(dataset_path):
        dataset = ManiDataset(
            dataset_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=attack_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )
    else:
        # assume a JSON file path for JsonDataset
        dataset = JsonDataset(
            dataset_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=attack_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )

    if getattr(args, "distributed", False):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return loader


def main():
    args = get_args()

    # DDP init sets: args.rank / args.world_size / args.gpu / args.distributed
    misc.init_distributed_mode(args)

    # REPRO
    seed = 42 + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # model
    model = build_model(args, device)
    load_ckpt_strict(model, args.checkpoint_path, map_location=device)

    if getattr(args, "distributed", False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu] if torch.cuda.is_available() else None, find_unused_parameters=False
        )

    # robustness sweeps (Rotation only to avoid missing imports)
    robustness_list = [
        RotationWrapper([0]), # 表示原性能
        # GaussianBlurWrapper([3, 7, 11, 15, 19, 23]),
        RotationWrapper([2, 4, 6, 8, 10, 12]),
        GaussianNoiseWrapper([3, 7, 11, 15, 19, 23]), 
        JpegCompressionWrapper([50, 60, 70, 80, 90, 100]),
        ResolutionChangeWrapper([0.75, 0.65, 0.55, 0.45, 0.35, 0.25]),
        ColorJitterWrapper([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    ]

    # evaluator list
    evaluator_list = [PixelF1(threshold=args.threshold, mode="origin")]

    # read datasets json: [[name, path], ...]
    with open(args.test_data_json, "r") as f:
        datasets = json.load(f)
    assert isinstance(datasets, (list, tuple)) and len(datasets) > 0
    if misc.is_main_process():
        print(f"[DEBUG] datasets path: {datasets}")

    # run
    start_time = time.time()
    for wrapper in robustness_list:
        for attack_param, attack_transform in wrapper:
            if misc.is_main_process():
                print(f"\n[ROBUST TEST] {wrapper.__class__.__name__} param={attack_param}")

            # log dir per attack setting
            args.full_log_dir = os.path.join(
                args.log_dir, f"{wrapper.__class__.__name__}", str(attack_param)
            )
            log_writer = None
            if misc.is_main_process():
                os.makedirs(args.full_log_dir, exist_ok=True)
                log_writer = SummaryWriter(log_dir=args.full_log_dir)

            # test all datasets listed
            for i, (name, path) in enumerate(datasets, 1):
                if misc.is_main_process():
                    print(f"[DEBUG] [{i}/{len(datasets)}] loading "
                          f"{'JsonDataset' if os.path.isfile(path) else 'ManiDataset'}, path is {path}")

                loader = build_loader(args, path, name, attack_transform)

                # run one epoch
                test_stats = test_one_epoch(
                    model=model,
                    data_loader=loader,
                    evaluator_list=evaluator_list,
                    device=device,
                    epoch=0,
                    log_writer=log_writer,
                    args=args
                )

                # log to file
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': int(attack_param), 'dataset': name}
                if misc.is_main_process():
                    with open(os.path.join(args.full_log_dir, "log.txt"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

            # flush TB
            if misc.is_main_process() and log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print(f"Total testing time: {total_time_str}")


if __name__ == "__main__":
    if not os.path.isdir(IMDL_LOCAL):
        raise RuntimeError(f"Local IMDLBenCo path not found: {IMDL_LOCAL}")
    Path("./robust_out").mkdir(parents=True, exist_ok=True)
    main()
