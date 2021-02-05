import os
from collections import OrderedDict
from typing import Tuple, List, Dict, Any, Callable, TypeVar, Union, Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ...base_architecture import Head, Model
from .configuration_solov1 import Solov1Config
from ..backbone_resnet import resnet101, resnet50
from ..neck_fpn import FeaturePyramidNetwork


class Solov1PredictNeck(FeaturePyramidNetwork):
    def __init__(
        self,
        config: Solov1Config = None,
        channels: Sequence[int] = [256, 512, 1024, 2048],
        selected_layers: Sequence[int] = [0, 1, 2, 3],
        fpn_channels: int = 256,
        extra_layers: bool = False,
        num_extra_fpn_layers: int = 1
    ) -> None:
        super().__init__(
            config,
            channels,
            selected_layers,
            fpn_channels,
            extra_layers,
            num_extra_fpn_layers
        )


class CategoryLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        num_groups,
    ) -> None:
        super().__init__(OrderedDict([
            ('conv', nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias)),
            ('gn', nn.GroupNorm(num_groups, out_channels)),
            ('relu', nn.ReLU())
        ]))


class InstanceLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        num_groups,
    ) -> None:
        super().__init__(OrderedDict([
            ('conv', nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias)),
            ('gn', nn.GroupNorm(num_groups, out_channels)),
            ('relu', nn.ReLU())
        ]))


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

        self.category_layers = nn.ModuleList()
        self.instance_layers = nn.ModuleList()
        for i in range(self.num_head_layers):
            if i == 0:
                in_channels = self.in_channels + 2
            else:
                in_channels = self.fpn_channels

            self.instance_layers.append(
                InstanceLayer(
                    in_channels,
                    self.fpn_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
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
                    bias=False,
                    num_groups=32
                )
            )

        self.pred_instance_layers = nn.ModuleList()
        for grid in self.grids:
            self.pred_instance_layers.append(
                nn.Conv2d(self.fpn_channels, grid**2, kernel_size=1)
            )

        self.pred_category_layer = nn.Conv2d(
            self.fpn_channels, self.num_classes, kernel_size=3, padding=1
        )

    def forward(self, inputs: List[Tensor]):
        inputs = self.split_feature_maps(inputs)
        feature_map_sizes = [feature_map.size()[-2:] for feature_map in inputs]
        # print(feature_map_sizes)
        upsampled_size = \
            (feature_map_sizes[0][0] * 2, feature_map_sizes[0][1] * 2)

        pred_masks, pred_labels = \
            self.partial_apply(
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
                align_corners=False),#, recompute_scale_factor=True),
            inputs[1],
            inputs[2],
            inputs[3],
            F.interpolate(
                inputs[4], size=inputs[3].shape[-2:],
                mode='bilinear')#, align_corners=False)
        )

    # def split_feature_maps(self, inputs: List[Tensor]) -> Tuple[Tensor]:
    #     """
    #     Returns:
    #     """
    #     return (
    #         F.interpolate(
    #             inputs[0], scale_factor=0.5, mode='bilinear',
    #             align_corners=False),#, recompute_scale_factor=True),
    #         inputs[1],
    #         inputs[2],
    #         inputs[3],
    #         F.interpolate(
    #             inputs[4], size=inputs[3].shape[-2:],
    #             mode='bilinear')#, align_corners=False)
    #     )

    def forward_single(self, inputs, idx, upsampled_size: Tuple = None):
        instances = inputs
        categories = inputs
        print('instance size()', instances.size())  # [B, C, H, W]

        x_range = torch.linspace(-1, 1, instances.size(3), device=instances.device)
        y_range = torch.linspace(-1, 1, instances.size(2), device=categories.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([instances.size(0), 1, -1, -1])
        x = x.expand([instances.size(0), 1, -1, -1])
        coords = torch.cat([x, y], dim=1)
        instances = torch.cat([instances, coords], 1)

        for i, ins_layer in enumerate(self.instance_layers):
            instances = ins_layer(instances)

        instances = F.interpolate(instances, scale_factor=2.0, mode='bilinear')#, align_corners=False)
        pred_masks = self.pred_instance_layers[idx](instances)

        # print('pred_masks', pred_masks.size())
        # test_seg_masks = pred_masks[0, :, :, 0] > 0.5 # cfg.mask_thr
        # test_masks = test_seg_masks.detach().cpu().numpy()[0] * 255
        # print(test_masks.shape)
        # # test_masks = test_masks.transpose(1, 2, 0)
        # import cv2
        # cv2.imwrite('solo-test-forward.jpg', test_masks)

        for i, cate_layer in enumerate(self.category_layers):
            if i == self.cate_down_pos:
                seg_num_grid = self.grids[idx]
                categories = F.interpolate(categories, size=seg_num_grid, mode='bilinear')#, align_corners=False)
            categories = cate_layer(categories)

        pred_labels = self.pred_category_layer(categories)
        if self.training:
            return pred_masks, pred_labels
        else:
            pred_masks = F.interpolate(pred_masks.sigmoid(), size=upsampled_size, mode='bilinear')
            # print(pred_masks.size())
            pred_labels = points_nms(pred_labels.sigmoid(), kernel=2).permute(0, 2, 3, 1)

        return pred_masks, pred_labels


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


class Solov1Pretrained(Model):
    config_class = Solov1Config
    base_model_prefix = 'solov1'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = Solov1Model(config)

        return model


class Solov1Model(Solov1Pretrained):
    """
    """
    model_name = 'solov1'

    def __init__(
        self,
        config: Solov1Config,
        backbone=None,
        neck=None,
        head=None,
        fpn_channels=256,
    ) -> None:
        super().__init__(config)
        self.config = config

        if backbone is None:
            self.backbone = resnet50()

        self.neck = Solov1PredictNeck(
            config, self.backbone.channels, extra_layers=False, num_extra_fpn_layers=1)

        self.head = Solov1PredictHead(config)

    def forward(self, inputs):
        inputs = self.check_inputs(inputs)
        # self.config.device = inputs.device

        # self.config.size = (inputs.size(2), inputs.size(3))

        outputs = self.backbone(inputs)
        # outputs = [outputs[i] for i in self.config.selected_layers]
        outputs = self.neck(outputs)

        outputs = self.head(outputs)

        # if self.training:
        #     pred_instances = F.interpolate(pred_instances.sigmoid(), size=upsampled_size, mode='bilinear', align_corners=False)
        #     pred_categories = points_nms(pred_categories.sigmoid(), kernel=2).permute(0, 2, 3, 1)

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

    def load_weights(self, path):
        import re

        try:
            state_dict = torch.load(path)['state_dict']
        except KeyError:
            state_dict = torch.load(path)

        numbering = {str(k): str(v) for k, v in enumerate([3, 2, 1, 0])}
        # numbering2 = {str(k): str(v) for k, v in enumerate([4, 3, 2, 1, 0])}
        for key in list(state_dict.keys()):
            p = key.split('.')
            if p[0] == 'backbone':
                if p[1].startswith('layer'):
                    i = int(p[1][5:])-1
                    if p[3].startswith('conv'):
                        new_key = f'backbone.layers.{i}.{p[2]}.{p[3]}.0.{p[4]}'
                        state_dict[new_key] = state_dict.pop(key)
                    elif p[3] == 'downsample':
                        new_key = f'backbone.layers.{i}.{p[2]}.{p[3]}.{p[4]}.{p[5]}'
                        state_dict[new_key] = state_dict.pop(key)
                    else:
                        new_key = f'backbone.layers.{i}.{p[2]}.{p[3]}.{p[4]}'
                        state_dict[new_key] = state_dict.pop(key)
                else:
                    p1 = re.findall('[a-zA-Z]+', p[1])[0]
                    new_key = f'backbone.{p1}.{p[2]}'
                    state_dict[new_key] = state_dict.pop(key)

            elif p[0] == 'neck':
                if p[1] == 'lateral_convs':
                    i = numbering.get(p[2])
                    new_key = f'neck.lateral_layers.{i}.{p[4]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'fpn_convs':
                    i = numbering.get(p[2])
                    new_key = f'neck.predict_layers.{i}.{p[4]}'
                    state_dict[new_key] = state_dict.pop(key)

            elif p[0] == 'bbox_head':
                if p[1] == 'ins_convs':
                    new_key = f'head.instance_layers.{p[2]}.{p[3]}.{p[4]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'cate_convs':
                    new_key = f'head.category_layers.{p[2]}.{p[3]}.{p[4]}'
                    state_dict[new_key] = state_dict.pop(key)
                # elif p[1] == 'solo_ins_list':
                #     new_key = f'head.pred_instance_layers.{numbering2.get(p[2])}.{p[3]}'
                #     state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'solo_ins_list':
                    new_key = f'head.pred_instance_layers.{p[2]}.{p[3]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'solo_cate':
                    new_key = f'head.pred_category_layer.{p[2]}'
                    state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(state_dict)
