#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from collections import deque

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from utils.util_image import ImageSpliterTh
from models.spynet import flow_warp

class BaseSampler:
    def __init__(
            self,
            configs,
            sf=None,
            use_fp16=False,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            desired_min_size=64,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_fp16 = use_fp16
        self.desired_min_size = desired_min_size
        if sf is None:
            sf = configs.diffusion.params.sf
        self.sf = sf

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, 'Please assign one available GPU using CUDA_VISIBLE_DEVICES!'

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        if self.use_fp16:
            model.dtype = torch.float16
            model.convert_to_fp16()
        self.model = model.eval()

        # flownet
        flownet =  util_common.instantiate_from_config(self.configs.flownet).cuda()
      #  ckpt_path = self.configs.flownet.ckpt_path
      #  assert ckpt_path is not None
     #   self.load_model(flownet, ckpt_path)
        self.flownet = flownet.eval()

        # fuse
        #fuse = util_common.instantiate_from_config(self.configs.fuse).cuda()
    #    ckpt_path = self.configs.fuse.ckpt_path
    #    assert ckpt_path is not None
    #    self.load_model(fuse, ckpt_path)
        self.fuse = None

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half()
            else:
                self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

class FSDMSampler(BaseSampler):
    def sample_func(self, y0, pre_im_sr_pch=None, noise_repeat=False,one_step=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        desired_min_size = self.desired_min_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
            if pre_im_sr_pch is not None:
                pre_im_sr_pch = F.pad(pre_im_sr_pch, pad=( 0, pad_w*self.sf, 0, pad_h*self.sf), mode='reflect')
        else:
            flag_pad = False

        model_kwargs={'lq':y0,} if self.configs.model.params.cond_lq else None
        results = self.base_diffusion.p_sample_loop(
                y=pre_im_sr_pch,
                model=self.model,
                first_stage_model=self.autoencoder,
                fuse_model=self.fuse,
                flow=None,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                one_step=one_step
                )    # This has included the decoding for latent space

        if flag_pad:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]

        return results.clamp_(-1.0, 1.0)

    def inference(self, in_path, out_path, bs=1, noise_repeat=False,window_size=3,one_step=False):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        def caculate_mask(im_lq_tensor, pre_im_sr_tensor=None):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [0,1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.sf,
                        extra_bs=self.chop_bs,
                        extra_im=pre_im_sr_tensor,
                        )
                for im_lq_pch, index_infos, pre_im_sr_pch in im_spliter:
                    # print(im_lq_pch.shape)
                    im_sr_pch = self.sample_func(
                            (im_lq_pch - 0.5) / 0.5,
                            pre_im_sr_pch = (pre_im_sr_pch - 0.5) / 0.5 if pre_im_sr_pch is not None else None,
                            noise_repeat=noise_repeat,
                            )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                        (im_lq_tensor - 0.5) / 0.5,
                        pre_im_sr_pch = (pre_im_sr_tensor - 0.5) / 0.5 if pre_im_sr_tensor is not None else None,
                        noise_repeat=noise_repeat,
                        )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor
        
        def _process_per_image(im_lq_tensor, pre_im_sr_tensor=None):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [0,1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.sf,
                        extra_bs=self.chop_bs,
                        extra_im=pre_im_sr_tensor,
                        )
                for im_lq_pch, index_infos, pre_im_sr_pch in im_spliter:
                    # print(im_lq_pch.shape)
                    im_sr_pch = self.sample_func(
                            (im_lq_pch - 0.5) / 0.5,
                            pre_im_sr_pch = (pre_im_sr_pch - 0.5) / 0.5 if pre_im_sr_pch is not None else None,
                            noise_repeat=noise_repeat,
                            one_step=one_step,
                            )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                        (im_lq_tensor - 0.5) / 0.5,
                        pre_im_sr_pch = (pre_im_sr_tensor - 0.5) / 0.5 if pre_im_sr_tensor is not None else None,
                        noise_repeat=noise_repeat,
                        one_step=one_step,
                        )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
        if not out_path.exists():
            out_path.mkdir(parents=True)

        if bs > 1:
            assert in_path.is_dir(), "Input path must be folder when batch size is larger than 1."

            data_config = {'type': 'folder',
                           'params': {'dir_path': str(in_path),
                                      'transform_type': 'default',
                                      'transform_kwargs': {
                                          'mean': 0.0,
                                          'std': 1.0,
                                          },
                                      'need_path': True,
                                      'recursive': True,
                                      'length': None,
                                      }
                           }
            dataset = create_dataset(data_config)
        #    self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    )
            # TODO
            for micro_data in dataloader:
                results = _process_per_image(micro_data['lq'].cuda())    # b x h x w x c, [0, 1], RGB

                for jj in range(results.shape[0]):
                    im_sr = util_image.tensor2img(results[jj], rgb2bgr=True, min_max=(0.0, 1.0))
                    im_name = Path(micro_data['path'][jj]).stem
                    im_path = out_path / f"{im_name}.png"
                    util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
        else:
            if not in_path.is_dir():
                im_lq = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
                im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w
                im_sr_tensor = _process_per_image(im_lq_tensor)
                im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))

                im_path = out_path / f"{in_path.stem}.png"
                util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
            else:
                im_path_list = [x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")]
           #     self.write_log(f'Find {len(im_path_list)} images in {in_path}')
                print(window_size)
                pre_im_lq_tensors = deque(maxlen=window_size)
                pre_im_sr_tensors = deque(maxlen=window_size)
                self.base_diffusion.clear_cache()
                for i, im_path in enumerate(sorted(im_path_list)):
                    print(im_path)
                    im_lq = util_image.imread(im_path, chn='rgb', dtype='float32')  # h x w x c
                    im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w
                    up_im_lq_tensor = F.interpolate(im_lq_tensor, scale_factor=self.sf,
                                                    mode='bicubic')
                    if len(pre_im_sr_tensors)>0:
                  #  if False:
                        flow_masks= []
                        warp_pre_im_sr_tensors = []
                        for j,(pre_im_lq_tensor, pre_im_sr_tensor) in enumerate(zip(pre_im_lq_tensors, pre_im_sr_tensors),1):
                            with torch.no_grad():
                                forward_flow = self.flownet(im_lq_tensor, pre_im_lq_tensor)
                            forward_flow = torch.nn.functional.interpolate(forward_flow * self.sf,
                                                                           scale_factor=self.sf).permute(0, 2,
                                                                                                   3, 1)
                            with torch.no_grad():
                                
                                warp_pre_im_sr_tensor = flow_warp(pre_im_sr_tensor, forward_flow)
                            
                            warp_pre_im_sr_tensors.append(warp_pre_im_sr_tensor)

                            backward_warp_flow = self.flownet(pre_im_sr_tensor, warp_pre_im_sr_tensor).permute(0, 2,
                                                                                                 3, 1)
                            flow_consistant_err = torch.norm(forward_flow + backward_warp_flow, p=2, dim=-1) 
                            flow_mask = (-0.1*flow_consistant_err).exp()
                            flow_mask[flow_mask < 0.05] = 0
                            flow_masks.append(flow_mask)
                        if len(flow_masks)==1:
                            #print("in this")
                            warp_pre_im_sr_tensor = up_im_lq_tensor * (1-flow_masks[0]) + flow_masks[0] * warp_pre_im_sr_tensors[0]
                        else:
                            warp_pre_im_sr_tensor=0
                            for i in range(len(flow_masks)):
                                weight=0
                                if i==len(flow_masks)-1:
                                    weight=1/(2**i)
                                else:
                                    weight=1/(2**(i+1)) 
                                warp_pre_im_sr_tensor += weight*(up_im_lq_tensor * (1-flow_masks[-(i+1)]) + flow_masks[-(i+1)] * warp_pre_im_sr_tensors[-(i+1)])
                    else:
                        warp_pre_im_sr_tensor = up_im_lq_tensor
                    im_sr_tensor = _process_per_image(im_lq_tensor, pre_im_sr_tensor=warp_pre_im_sr_tensor)
                    pre_im_lq_tensors.append(im_lq_tensor.clone().detach())
                    pre_im_sr_tensors.append(im_sr_tensor.clone().detach())
                    im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))

                    im_path = out_path / f"{im_path.stem}.png"
                    util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')

if __name__ == '__main__':
    pass

