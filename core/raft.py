import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import BasicUpdateBlock
from core.extractor import BasicEncoder
from core.corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.runner import load_checkpoint

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4
        self.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0)
        self.update_block = BasicUpdateBlock( hidden_dim=hdim)
            
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=True)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        print('#Params of {}: {:<.4f} [M]'.format("-------raft",
                                                  num_parameters / 10 ** 6))
        

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=True):
        """ Estimate optical flow between pair of frames """
        #print(image1.shape,image2.shape)
        pad=0
        if image1.shape[3]==64 and image1.shape[2]==64:
            pad=1
            padding = (96, 96, 96, 96)  # (left, right, top, bottom)
            image1 = F.pad(image1, padding, mode='replicate')
            image2 = F.pad(image2, padding, mode='replicate')
        elif image1.shape[2] % 8 != 0 or image1.shape[3] % 8 != 0:  # Check if image size is not divisible by 8
            pad = 2
            # Calculate padding needed to make dimensions divisible by 8
            pad_h = 8 - image1.shape[2] % 8
            pad_w = 8 - image1.shape[3] % 8
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)  # (left, right, top, bottom)
            image1 = F.pad(image1, padding, mode='replicate')
            image2 = F.pad(image2, padding, mode='replicate')

        image1 = 2 * (image1 ) - 1.0
        image2 = 2 * (image2 ) - 1.0
        

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=True):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        with autocast(enabled=True):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            #print(itr)
            coords1 = coords1.detach()
            #print(coords1.shape)
            corr = corr_fn(coords1) # index correlation volume
        

            flow = coords1 - coords0
            with autocast(enabled=True):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            #flow_predictions.append(flow_up)

        if test_mode:
            if pad==2:
                # Remove padding from the output flow_up
                flow_up = flow_up[:, :, pad_h // 2:image1.shape[2] -pad_h + pad_h // 2, pad_w // 2:image1.shape[3] - pad_w // 2]
            elif pad==1:
                flow_up = flow_up[:, :, 96:160, 96:160]
                
            return flow_up
            
        return flow_predictions
        
if __name__=='__main__':
    import cv2
    import numpy as np

    def pipeline(path):
        img = cv2.imread(path)[:, :, ::-1]
        img = img.astype(np.float32)/255.
        img = torch.from_numpy(img.transpose(2, 0, 1)).cuda()
        return img.unsqueeze(0)

    spynet = RAFT("/home/ysj/ResShiftV2/weights/raft-things.pth").cuda()
    img1 = pipeline('/media/amax/chenchao/REDS/train/train_sharp_bicubic/X4/000/00000001.png')
    img2 = pipeline('/media/amax/chenchao/REDS/train/train_sharp_bicubic/X4/000/00000002.png')
    with torch.no_grad():
        flows = spynet(img2, img1)
        img12 = flow_warp(img1, flows.permute(0, 2, 3, 1))
        
        print(img12.max())
        
        
    #diff = (img12 - img1).squeeze().abs().mean(dim=0).cpu().numpy()
  #  diff = np.clip(diff, 0, 1)
    #diff = diff / diff.max()

    img12 = img12.squeeze().cpu().numpy().transpose(1, 2, 0)* 255.
    #mask = np.ones(img12.shape[:2]) * 255
    #img12 = img12 * (1-diff[..., None]) + (mask * diff)[..., None]
    img12 = np.clip(img12, 0, 255)
    #print(diff)
    cv2.imwrite('tmp.png', img12[:, :, ::-1])
