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

from mmcls.models.backbones.shufflenet_v1 import ShuffleUnit
from mmcls.models.backbones.shufflenet_v2 import InvertedResidual


@BACKBONES.register_module()
class SECONDShufflenetV1(BaseModule):
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
                 groups = 4,
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
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1
                ),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
                ShuffleUnit(
                    out_channels[i],
                    out_channels[i],
                    groups = groups,
                    first_block = True,
                    combine = 'add'
                ),
            ]
            for j in range(layer_num - 1):
                block.append(
                    ShuffleUnit(
                        out_channels[i],
                        out_channels[i],
                        groups = groups,
                        first_block = False,
                        combine = 'add',
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
class SECONDShufflenetV2(BaseModule):
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
                 groups = 4,
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
                InvertedResidual(
                    in_filters[i],
                    out_channels[i],
                    stride = layer_strides[i],
                    conv_cfg = conv_cfg,
                    act_cfg = act_cfg,
                    norm_cfg = norm_cfg,
                ),
            ]
            for j in range(layer_num - 1):
                block.append(
                    InvertedResidual(
                        out_channels[i],
                        out_channels[i],
                        stride = 1,
                        conv_cfg = conv_cfg,
                        act_cfg = act_cfg,
                        norm_cfg = norm_cfg,
                    ),
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

