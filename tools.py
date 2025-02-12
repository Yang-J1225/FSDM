#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url


def get_configs(args):
    if args.task == 'realsrx4':
        configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
    elif args.task == 'bicsrx4_opencv':
        configs = OmegaConf.load('./configs/bicubic_swinunet_bicubic256.yaml')
    elif args.task == 'bicsrx4_matlab':
        configs = OmegaConf.load('./configs/bicubic_swinunet_bicubic256.yaml')
        configs.diffusion.params.kappa = 1
    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
         load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v1.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    #configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = args.steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    #assert(args.chop_size == 512)
    args.chop_size=512
    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    elif args.chop_size == 64:
        chop_stride = 56
    #chop_stride = args.chop_size - 8
    
    print(args.chop_size)
    return configs, chop_stride


