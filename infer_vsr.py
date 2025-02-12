#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import csv
import random
import argparse
import numpy as np
import torch
from utils import util_image
from tqdm import tqdm

from sampler import FSDMSampler
from tools import get_configs

def get_parser():
    parser = argparse.ArgumentParser(description="single_infer_sr")
    parser.add_argument("--input_path", type=str, default="testdata/Bicubicx4/lq_opencv/ILSVRC2012_val_00000079.png", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length.")
    parser.add_argument("-window_size", type=int, default=3, help="window_size.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--chop_size",
        type=int,
        default=512,
        choices=[512, 256],
        help="Chopping forward.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="bicsrx4_matlab",
        choices=['realsrx4', 'bicsrx4_opencv', 'bicsrx4_matlab'],
        help="Chopping forward.",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_parser()
    configs, chop_stride = get_configs(args)
    
    fsdm_sampler = FSDMSampler(
        configs,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
    )

    if os.path.exists(save_sr_dir := args.out_path):
        os.makedirs(save_sr_dir, exist_ok=True)
    with torch.no_grad():
        fsdm_sampler.inference(args.input_path, save_sr_dir, bs=1, noise_repeat=False,window_size = args.window_size)


if __name__ == '__main__':
    main()