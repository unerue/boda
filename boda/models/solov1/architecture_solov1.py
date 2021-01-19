import os
import math
import itertools
import functools
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Dict, Any, Callable, TypeVar, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ...base_architecture import Neck, Head, Model
from .configuration_solov1 import Solov1Config
from ..backbone_resnet import resnet101, resnet50
from ...utils.mask import points_nms


def multi_apply(func: Callable, *args, **kwargs) -> List[Tensor]:
    """Multiple apply

    Args:
        func (:obj:`Callable`):
    """
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class Solov1PredictNeck(Neck):
    def __init__(
        self,
        config: Solov1Config = None,
        channels: List[int] = [256, 512, 1024, 2048],
        fpn_channels: int = 256,
        num_extra_layers: int = 1
    ) -> None:
        super().__init__()
        self.config = config
        self.fpn_channels = fpn_channels
        self.num_extra_layers = num_extra_layers
        # TODO: update config method
        # self.update_config(config)

        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                self.fpn_channels,
                kernel_size=1
            ) for in_channels in reversed(channels)])

        self.predict_layers = nn.ModuleList([
            nn.Conv2d(
                self.fpn_channels,
                self.fpn_channels,
                kernel_size=3,
                padding=1) for _ in channels])

        if self.num_extra_layers > 0:
            self.extra_layers = nn.ModuleList([
                nn.Conv2d(
                    self.fpn_channels,
                    self.fpn_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1) for _ in range(self.num_extra_layers)])

    def forward(self, inputs: List[Tensor]):
        device = inputs[0].device

        x = torch.zeros(1, device=device)
        outputs = [x for _ in range(len(inputs))]
        i = len(inputs)
        for lateral_layer in self.lateral_layers:
            i -= 1
            if i < len(inputs) - 1:
                _, _, h, w = inputs[i].size()
                x = F.interpolate(
                    x, size=(h, w), mode='nearest')
            x = x + lateral_layer(inputs[i])
            outputs[i] = x

        i = len(inputs)
        for predict_layer in self.predict_layers:
            i -= 1
            outputs[i] = F.relu(predict_layer(outputs[i]))

        if self.num_extra_layers > 0:
            for extra_layer in self.extra_layers:
                outputs.append(extra_layer(outputs[-1]))
        print('out fpn', len(outputs))
        return outputs


class Solov1PredictHead(Head):
    def __init__(
        self,
        config: Solov1Config,
        in_channels: int = 256,
        fpn_channels: int = 256,
        num_head_layers: int = 7,
        grids: List = [40, 36, 24, 16, 12],
        strides: List = [4, 8, 16, 32, 64],
        base_edges: List = [16, 32, 64, 128, 256],
        scales: List = ((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
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

        self.category_layers = nn.ModuleList()
        self.instance_layers = nn.ModuleList()
        for i in range(self.num_head_layers):
            if i == 0:
                in_channels = self.in_channels + 2
            else:
                in_channels = self.fpn_channels

            self.instance_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        self.fpn_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True),
                    nn.GroupNorm(32, self.fpn_channels)))

            if i == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.fpn_channels
            self.category_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        self.fpn_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True),
                    nn.GroupNorm(32, self.fpn_channels)))

        self.solo_instances = nn.ModuleList()
        for grid in self.grids:
            self.solo_instances.append(
                nn.Conv2d(self.fpn_channels, grid**2, kernel_size=1)
            )
        self.solo_category = nn.Conv2d(
            self.fpn_channels, self.num_classes-1, kernel_size=3, padding=1
        )

    def forward(self, inputs: List[Tensor]):
        inputs = self.split_feature_maps(inputs)
        feature_map_sizes = [feature_map.size()[-2:] for feature_map in inputs]
        upsampled_size = \
            (feature_map_sizes[0][0] * 2, feature_map_sizes[0][1] * 2)

        pred_instances, pred_categories = \
            multi_apply(
                self.forward_single,
                inputs,
                list(range(len(self.grids))),
                upsampled_size=upsampled_size)

        return pred_instances, pred_categories

    def split_feature_maps(self, inputs: List[Tensor]) -> Tuple[Tensor]:
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
        pred_instances = self.solo_instances[idx](instances)

        for i, cate_layer in enumerate(self.category_layers):
            if i == self.cate_down_pos:
                seg_num_grid = self.grids[idx]
                categories = F.interpolate(categories, size=seg_num_grid, mode='bilinear', align_corners=False)
            categories = cate_layer(categories)

        pred_categories = self.solo_category(categories)
        if self.training:
            pred_instances = F.interpolate(pred_instances.sigmoid(), size=upsampled_size, mode='bilinear', align_corners=False)
            pred_categories = points_nms(pred_categories.sigmoid(), kernel=2).permute(0, 2, 3, 1)

        return pred_instances, pred_categories


class Solov1Pretrained(Model):
    config_class = Solov1Config
    base_model_prefix = 'solov1'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = Solov1Model(config)
        # model.state_dict(torch.load('test.pth'))
        return model

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Solov1Model(Solov1Pretrained):
    """
    """
    model_name = 'solov1'

    def __init__(
        self,
        config: Solov1Config,
        backbone=None, neck=None, head=None,
        num_features=256,
        out_channels_fpn=256,
        num_grids=5,

    ) -> None:
        super().__init__(config)
        self.config = config

        if backbone is None:
            self.backbone = resnet50()

        self.neck = Solov1PredictNeck(
            config, self.backbone.channels)

        self.head = Solov1PredictHead(config)

    def forward(self, inputs):
        inputs = self.check_inputs(inputs)
        # self.config.device = inputs.device

        # self.config.size = (inputs.size(2), inputs.size(3))

        outputs = self.backbone(inputs)
        print(len(outputs))
        for o in outputs:
            print(o.size())
        # outputs = [outputs[i] for i in self.config.selected_layers]
        outputs = self.neck(outputs)
        for o in outputs:
            print(o.size())
        print('neck', len(outputs))

        outputs = self.head(outputs)

        # for o in outputs:
        #     print(o.size())

        # proto_input = outputs[0]
        # # print(proto_input.size())
        # segout = self.semantic_layer(outputs[0])
        # proto_output = self.proto_net(proto_input)
        # for i, layer in zip(self.config.selected_layers, self.head_layers):
        #     print(i, layer)
        #     print(outputs[i].size())
        #     output = layer(outputs[i])

        return outputs