# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn.bricks import DepthwiseSeparableConvModule as DWSConv
from mmcv.runner import BaseModule
from torch import nn as nn
import torch.nn.functional as F
import torch

from mmdet.models import BACKBONES

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        BIAS = True
        RELU_INPLACE = True
        
        reduction = 0.5
        if 2 == stride:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
        
        self.bconv1 = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=BIAS),
            nn.BatchNorm2d(int(in_channels * reduction)),
            nn.ReLU(inplace=RELU_INPLACE),
        )
        self.bconv2 = nn.Sequential(
            nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=BIAS),
            nn.BatchNorm2d(int(in_channels * reduction * 0.5)),
            nn.ReLU(inplace=RELU_INPLACE),
        )
        self.bconv3 = nn.Sequential(
            nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=BIAS),
            nn.BatchNorm2d(int(in_channels * reduction)),
            nn.ReLU(inplace=RELU_INPLACE),
        )
        self.bconv4 = nn.Sequential(
            nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=BIAS),
            nn.BatchNorm2d(int(in_channels * reduction)),
            nn.ReLU(inplace=RELU_INPLACE),
        )
        self.bconv5 = nn.Sequential(
            nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=BIAS),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=RELU_INPLACE),
        )
        self.shortcut = nn.Sequential(
            nn.ReLU(inplace=False),
        )
        if 2 == stride or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, stride, bias=BIAS),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=RELU_INPLACE),
            )

        self.out_block = nn.ReLU(inplace=False)
            
    def forward(self, x):
        input = x
        x = self.bconv1(x)
        x = self.bconv2(x)
        x = self.bconv3(x)
        x = self.bconv4(x)
        x = self.bconv5(x)
        input = self.shortcut(input)
        x = x + input
        x = self.out_block(x)
        return x
    

@BACKBONES.register_module()
class SECONDSqueezeNext(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None,
                 pretrained=None):
        super().__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                BasicBlock(
                    in_filters[i],
                    out_channels[i], 
                    stride = layer_strides[i],
                )
            ]
            for j in range(layer_num):
                block.append(
                    BasicBlock(
                        out_channels[i],
                        out_channels[i], 
                        stride = 1,
                    )
                )

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)
