# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.cnn.bricks import DepthwiseSeparableConvModule as DWSConv
from mmcv.runner import BaseModule
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models import BACKBONES
from mmcv.runner import BaseModule, Sequential
from torch import nn as nn

import timm

from timm.models.cspnet import DarkBlock, DarkStage, CrossStage

cspdarknet53=dict(
    stem=dict(out_chs=32, kernel_size=3, stride=1, pool=''),
    stage=dict(
        out_chs=(64, 128, 256, 512, 1024),
        depth=(1, 2, 8, 8, 4),
        stride=(2,) * 5,
        exp_ratio=(2.,) + (1.,) * 4,
        bottle_ratio=(0.5,) + (1.0,) * 4,
        block_ratio=(1.,) + (0.5,) * 4,
        down_growth=True,
    )
),
darknet53=dict(
    stem=dict(out_chs=32, kernel_size=3, stride=1, pool=''),
    stage=dict(
        out_chs=(64, 128, 256, 512, 1024),
        depth=(1, 2, 8, 8, 4),
        stride=(2,) * 5,
        bottle_ratio=(0.5,) * 5,
        block_ratio=(1.,) * 5,
    )
)

@BACKBONES.register_module()
class SECONDDarknet(BaseModule):
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
                 bottle_ratio=[0.5, 0.5, 0.5],
                 block_ratio=[1.0, 1.0, 1.0],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d'),
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
                DarkStage(
                    in_chs  = in_filters[i],
                    out_chs = out_channels[i],
                    stride  = layer_strides[i],
                    dilation = 1,
                    depth = layer_num,
                    block_ratio = block_ratio[i],
                    bottle_ratio = bottle_ratio[i],
                    groups = 1,
                    block_fn = DarkBlock,
                    act_layer = nn.LeakyReLU,
                    norm_layer = nn.BatchNorm2d,
                )
            ]
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


@BACKBONES.register_module()
class SECONDCSPDarknet(BaseModule):
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
                 exp_ratio=[2.0, 1.0, 1.0],
                 bottle_ratio=[0.5, 1.0, 1.0],
                 block_ratio=[1.0, 0.5, 0.5],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d'),
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
                CrossStage(
                    in_chs  = in_filters[i],
                    out_chs = out_channels[i],
                    stride  = layer_strides[i],
                    dilation = 1,
                    depth = layer_num,
                    exp_ratio = exp_ratio[i],
                    block_ratio = block_ratio[i],
                    bottle_ratio = bottle_ratio[i],
                    groups = 1,
                    block_fn = DarkBlock,
                    down_growth = True,
                    act_layer = nn.LeakyReLU,
                    norm_layer = nn.BatchNorm2d,
                )
            ]
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
