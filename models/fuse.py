import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .spynet import flow_warp
from .basic_ops import timestep_embedding
from .unet import ResBlock, TimestepEmbedSequential, AttentionBlock

class HighFrequencyFocus(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.mid_ch = mid_ch
        self.mask_conv = nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.residual_conv = nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.in_conv =  nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.out_conv =  nn.Conv2d(mid_ch, in_ch, 3, 1, 1)

        nn.init.normal_(self.out_conv.weight, 0, 1e-4)
        nn.init.constant_(self.out_conv.bias, 0)


    def forward(self, x):
        residual = x - F.interpolate(F.interpolate(x,
                                    size=(x.shape[-2]//2, x.shape[-1]//2),
                                    mode='bilinear'),
                                     size=x.shape[-2:],
                                     mode='bilinear')
        mask = (self.mask_conv(x) + self.residual_conv(residual)).sigmoid()
        return self.out_conv(self.in_conv(x) * mask)




class Fuse(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()

    def forward(self, cur_x,  t, pre_x=None, flow=None):
        return cur_x


