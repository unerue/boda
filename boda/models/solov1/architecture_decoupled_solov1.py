import os
import math
import itertools
import functools
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Dict, Any, Callable, TypeVar, Union, Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ...base_architecture import Neck, Head, Model
from .configuration_solov1 import Solov1Config
from ..backbone_resnet import resnet101, resnet50
from ...utils.mask import points_nms
from ..neck_fpn import FeaturePyramidNetwork
from .architecture_solov1 import InstanceLayer, CategoryLayer, Solov1PredictNeck, Solov1PredictHead, Solov1Model


class DecoupledSolov1PredictHead(Solov1PredictHead):
    def __init__(
        self,
        config: Solov1Config,
        in_channels: int = 256,
        fpn_channels: int = 256,
        num_head_layers: int = 7,
        grids: List = [40, 36, 24, 16, 12],
        strides: List = [4, 8, 16, 32, 64],
        base_edges: List = [16, 32, 64, 128, 256],
        scales: List = [[8, 32], [16, 64], [32, 128], [64, 256], [128, 512]],
        num_classes: int = 80
    ) -> None:
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.fpn_channels = fpn_channels
        self.num_head_layers = num_head_layers
        self.grids = grids
        self.strides = strides
        self.base_edges = base_edges
        self.scales = scales
        self.num_classes = num_classes

        self.cate_down_pos = 0

        delattr(self, 'instance_layers')

        self.x_instance_layers = nn.ModuleList()
        self.y_instance_layers = nn.ModuleList()
        self.category_layers = nn.ModuleList()
        for i in range(self.num_head_layers):
            if i == 0:
                in_channels = self.in_channels + 1
            else:
                in_channels = self.fpn_channels

            self.x_instance_layers.append(
                InstanceLayer(
                    in_channels,
                    self.fpn_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    num_groups=32
                )
            )

            self.y_instance_layers.append(
                InstanceLayer(
                    in_channels,
                    self.fpn_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    num_groups=32
                )
            )

            if i == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.fpn_channels

            self.category_layers.append(
                CategoryLayer(
                    in_channels,
                    self.fpn_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    num_groups=32
                )
            )

        self.x_decoupled_instance_layers = nn.ModuleList()
        self.y_decoupled_instance_layers = nn.ModuleList()
        self.pred_instance_layers = nn.ModuleList()
        for grid in self.grids:
            self.x_decoupled_instance_layers.append(
                nn.Conv2d(self.fpn_channels, grid, kernel_size=3, padding=1)
            )
            self.y_decoupled_instance_layers.append(
                nn.Conv2d(self.fpn_channels, grid, kernel_size=3, padding=1)
            )

        self.pred_category_layer = nn.Conv2d(
            self.fpn_channels, self.num_classes-1, kernel_size=3, padding=1
        )

    def forward(self, inputs: List[Tensor]):
        inputs = self.split_feature_maps(inputs)
        feature_map_sizes = [feature_map.size()[-2:] for feature_map in inputs]
        upsampled_size = \
            (feature_map_sizes[0][0] * 2, feature_map_sizes[0][1] * 2)

        pred_masks, pred_labels = \
            self.multi_apply(
                self.forward_single,
                inputs,
                list(range(len(self.grids))),
                upsampled_size=upsampled_size)

        return pred_masks, pred_labels

    def split_feature_maps(self, inputs: List[Tensor]) -> Tuple[Tensor]:
        """
        Returns:
        """
        return (
            F.interpolate(
                inputs[0], scale_factor=0.5, mode='bilinear',
                align_corners=False, recompute_scale_factor=True),
            inputs[1],
            inputs[2],
            inputs[3],
            F.interpolate(
                inputs[4], size=inputs[3].shape[-2:],
                mode='bilinear', align_corners=False)
        )

    def forward_single(self, inputs, idx, upsampled_size: Tuple = None):
        instances = inputs
        categories = inputs

        x_range = torch.linspace(-1, 1, instances.shape[-1], device=instances.device)
        y_range = torch.linspace(-1, 1, instances.shape[-2], device=categories.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([instances.shape[0], 1, -1, -1])
        x = x.expand([instances.shape[0], 1, -1, -1])
        coords = torch.cat([x, y], 1)
        instances = torch.cat([instances, coords], 1)

        for i, ins_layer in enumerate(self.instance_layers):
            instances = ins_layer(instances)

        instances = F.interpolate(instances, scale_factor=2.0, mode='bilinear', align_corners=False)
        pred_masks = self.pred_instance_layers[idx](instances)

        for i, cate_layer in enumerate(self.category_layers):
            if i == self.cate_down_pos:
                seg_num_grid = self.grids[idx]
                categories = F.interpolate(categories, size=seg_num_grid, mode='bilinear', align_corners=False)
            categories = cate_layer(categories)

        pred_labels = self.pred_category_layer(categories)

        return pred_masks, pred_labels
