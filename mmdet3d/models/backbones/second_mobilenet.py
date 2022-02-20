# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn.bricks import DepthwiseSeparableConvModule as DWSConv
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models import BACKBONES
from ..utils import InvertedResidual
"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 dw_norm_cfg='default',
                 dw_act_cfg='default',
                 pw_norm_cfg='default',
                 pw_act_cfg='default',
                 **kwargs):
"""


@BACKBONES.register_module()
class SECONDMobilenetV1(BaseModule):
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
        super(SECONDMobilenetV1, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                DWSConv(
                    in_filters[i],
                    out_channels[i],
                    kernel_size = 3,
                    stride      = layer_strides[i],
                    padding     = 1,
                    norm_cfg    = norm_cfg,
                    act_cfg     = act_cfg,
                ),
            ]
            for j in range(layer_num):
                block.append(
                    DWSConv(
                        out_channels[i],
                        out_channels[i],
                        kernel_size = 3,
                        stride      = 1,
                        padding     = 1,
                        norm_cfg    = norm_cfg,
                        act_cfg     = act_cfg,
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


@BACKBONES.register_module()
class SECONDMobilenetV2(BaseModule):
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
                 expand_ratio=6,
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
                InvertedResidual(
                    in_filters[i],
                    out_channels[i],
                    mid_channels     = int(round(in_filters[i] * expand_ratio)),
                    stride           = layer_strides[i],
                    with_expand_conv = expand_ratio != 1,
                    norm_cfg         = norm_cfg,
                    act_cfg          = act_cfg,
                ),
            ]
            for j in range(layer_num):
                block.append(
                    InvertedResidual(
                        out_channels[i],
                        out_channels[i],
                        mid_channels     = int(round(out_channels[i] * expand_ratio)),
                        stride           = 1,
                        with_expand_conv = expand_ratio != 1,
                        norm_cfg         = norm_cfg,
                        act_cfg          = act_cfg,
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
