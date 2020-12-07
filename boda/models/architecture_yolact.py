import os
import math
import itertools
from collections import defaultdict
from collections import OrderedDict
from typing import Tuple, List, Dict, Any, Callable, TypeVar

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import Neck, Head, Model
from .backbone_resnet import resnet101


class YolactPredictNeck(Neck):
    def __init__(self, config, in_channels) -> None:
        super().__init__()
        self.config = config

        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(
                _in_channels,
                self.config.num_features,
                kernel_size=1) for _in_channels in reversed(in_channels)])

        self.predict_layers = nn.ModuleList([
            nn.Conv2d(
                self.config.num_features,
                self.config.num_features,
                kernel_size=3,
                padding=self.config.padding) for _ in in_channels])

        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(
                self.config.num_features,
                self.config.num_features,
                kernel_size=3,
                stride=2,
                padding=1) for _ in range(self.config.num_downsamples)])

    def forward(self, inputs: List[Tensor]):
        outputs = []
        x = torch.zeros(1, device=self.config.device)
        for _ in range(len(inputs)):
            outputs.append(x)

        outputs = [x for _ in range(len(inputs))]

        j = len(inputs)
        for lateral_layer in self.lateral_layers:
            j -= 1
            if j < len(inputs) - 1:
                _, _, h, w = inputs[j].size()
                x = F.interpolate(
                    x, size=(h, w), mode='bilinear', align_corners=False)
            
            x = x + lateral_layer(inputs[j])
            outputs[j] = x

        j = len(inputs)
        for predict_layer in self.predict_layers:
            j -= 1
            outputs[j] = F.relu(predict_layer(outputs[j]))

        for downsample_layer in self.downsample_layers:
            outputs.append(downsample_layer(outputs[-1]))

        return outputs


# T = TypeVar('T', bound=Callable[..., Any])
def prior_cache(func):
    cache = defaultdict()
    def wrapper(*args):
        k, v = func(*args)
        if k not in cache:
            cache[k] = v
        return k, cache[k]
    return wrapper


class PriorBox:
    def __init__(self, priors, device) -> None:
        self.priors = priors
        self.device = device
        self.last_img_size = None
        self.last_conv_size = None
        pass

    @prior_cache
    def generate(self, conv_h, conv_w):
        size = (conv_h, conv_w)
        prior_data = []
        # Iteration order is important (it has to sync up with the convout)
        for j, i in itertools.product(range(conv_h), range(conv_w)):
            # +0.5 because priors are in center-size notation
            x = (i + 0.5) / conv_w
            y = (j + 0.5) / conv_h

            for ars in self.aspect_ratios:
                for scale in self.scales:
                    for ar in ars:
                        if not cfg.backbone.preapply_sqrt:
                            ar = sqrt(ar)

                        if cfg.backbone.use_pixel_scales:
                            w = scale * ar / cfg.max_size
                            h = scale / ar / cfg.max_size
                        else:
                            w = scale * ar / conv_w
                            h = scale / ar / conv_h

                        # This is for backward compatability with a bug where I made everything square by accident
                        if cfg.backbone.use_square_anchors:
                            h = w

                        prior_data += [x, y, w, h]

        self.priors = torch.Tensor(prior_data, device=device).view(-1, 4).detach()
        self.priors.requires_grad = False
        self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)
        self.last_conv_size = (conv_w, conv_h)

        return (conv_h, conv_w), prior_data
        # prior_cache[size] = None

# [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]


class ProtoNet(nn.Sequential):
    def __init__(self, config, in_channels, layers, include_last_relu=True) -> None:
        self.config = config
        
        mask_layers = OrderedDict()
        for i, v in enumerate(layers, 1):
            if isinstance(v[0], int):
                mask_layers[f'protonet{i}'] = nn.Conv2d(
                    in_channels, v[0], kernel_size=v[1], **v[2])

            elif v[0] is None:
                mask_layers[f'protonet{i}'] = nn.Upsample(
                    scale_factor=-v[1], mode='bilinear', align_corners=False, **v[2])
            
        if include_last_relu:
            mask_layers[f'relu{len(mask_layers)+1}'] = nn.ReLU()

        super().__init__(mask_layers)
        print(self)
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


class YolactPredictHead(Head):
    def __init__(self, config, in_channels, out_channels, aspect_ratios, scales, parent, index) -> None:
        super().__init__()
        self.config = config
        self.mask_dim = self.config.mask_dim
        self.num_priors = sum(len(x)*len(scales) for x in aspect_ratios)
        self.parent = [parent]
        self.out_channels = in_channels

        # 추후 extra_layers와 밑에 bbox_layer가 같이 통합된 메서드 제작
        # self.bbox_extra_layers, self.bbox_extra_layers, self.config.extra_layers = [
        #     self._add_extra_layer(out_channels, num_layers) for num_layers in self.config.extra_layers]

        # self.bbox_layer = nn.Conv2d(
        #     out_channels, self.num_priors * 4, kernel_size=3, padding=1)
        # self.conf_layer = nn.Conv2d(
        #     out_channels, self.num_prios * self.config.num_classes, kernel_size=3, padding=1)
        # self.mask_layer = nn.Conv2d(
        #     out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.bbox_layer = self._add_predict_layer(
            self.config.extra_layers[0], out_channels, self.num_priors * 4)
        self.conf_layer = self._add_predict_layer(
            self.config.extra_layers[0], out_channels, self.num_prios * self.config.num_classes)
        self.mask_layer = self._add_predict_layer(
            self.config.extra_layers[0], out_channels, self.num_priors * self.mask_dim)

    def _add_predict_layer(self, num_extra_layers, in_channels, out_channels):
        if num_extra_layers == 0:
            predict_layers = []
        else:
            predict_layers = [[
                nn.Conv2d(
                    in_channels, 
                    in_channels, 
                    kernel_size=3,
                    padding=1),
                nn.ReLU(inplace=True)] for _ in range(num_extra_layers)]

        predict_layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

        return nn.Sequential(*predict_layers)

    def forward(self, inputs):
        # src = self if self.parent[0] is None else self.parent[0]

        h, w = inputs.size(2), inputs.size(3)

        bbox_outputs = self.bbox_layers(inputs)
        conf_outputs = self.conf_layers(inputs)
        mask_outputs = self.mask_layers(inputs)


class YolactBase(Model):
    def __init__(self):
        super().__init__()


class YolactModel(YolactBase):
    """
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝
    """
    model_name = 'yolact'
    
    def __init__(self, config, backbone=None, neck=None, head=None) -> None:
        super().__init__()
        self.config = config

        if backbone is None:
            self.backbone = resnet101()
            selected_layers = [1, 2, 3]
            # num_layers = max(cfg.selected_layers) + 1
            num_layers = max(selected_layers) + 1
            while len(self.backbone.layers) < num_layers:
                self.backbone.add_layer()


        # if self.config.freeze_bn:
        #     self.freeze_bn()

        self.config.mask_dim = self.config.mask_size**2

        in_channels = self.backbone.channels[self.config.proto_src]
        in_channels += self.config.num_grids

        if neck is None:
            self.neck = YolactPredictNeck(config, [self.backbone.channels[i] for i in self.config.selected_layers])
            selected_layers = list(range(len(self.config.selected_layers) + self.config.num_downsamples))
            neck_channels = [self.config.num_features] * len(selected_layers)

        num_grids = 0
        # in_channels = self.backbone.channels[self.config.proto_src]
        in_channels = self.config.num_features
        # in_channels += num_grids
        print(in_channels)
        self.proto_net = ProtoNet(config, in_channels, self.config.proto_net, include_last_relu=False)

        # self.head_layers = nn.ModuleList()
        # num_heads = len(self.config.selected_layers)

        # for i, j in enumerate(self.config.selected_layers):
        #     parent = None
        #     head_layer = YolactPredictHead(
        #         config, 
        #         neck_channels[j], 
        #         neck_channels[j],
        #         aspect_ratios=self.config.aspect_ratios[i],
        #         scales=self.config.scales[i],
        #         parent=parent,
        #         index=i)
        #     self.head_layers.append(head_layer)


    def forward(self, inputs):
        inputs = self.check_inputs(inputs)
        self.config.device = inputs.device

        outputs = self.backbone(inputs)
        print(self.backbone.channels)

        outputs = [outputs[i] for i in self.config.selected_layers]
        outputs = self.neck(outputs)
        print(len(outputs))
        for o in outputs:
            print(o.size())

        proto_input = outputs[0]
        print(proto_input.size())
        proto_output = self.proto_net(proto_input)
        print(len(proto_output))

        return outputs
