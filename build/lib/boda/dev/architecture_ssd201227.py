import os
import math
import functools
from collections import defaultdict
from typing import List, Dict, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import itertools 
from ..architecture_base import Neck, Head, Model
from .configuration_ssd import SsdConfig
from .backbone_vgg import vgg16


STRUCTURES = {
    'ssd300': [
        [('M', {'kernel_size': 3, 'stride': 1, 'padding': 1}), (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}), (1024, {'kernel_size': 1})],
        # [(1024, {'kernel_size': 1})],
        [(256, {'kernel_size': 1}), (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3})]],
    'ssd512': [
        [(256, {'kernel_size': 1}), (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride': 2, 'padding':  1})],
        [(128, {'kernel_size': 1})]]
}


class L2Norm(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        gamma: int = 10,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.gamma = gamma
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, inputs):
        norm_inputs = inputs.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        inputs = torch.div(inputs, norm_inputs)  # x /= norm_x
        outputs = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(inputs) * inputs
        return outputs


class SsdPredictNeck(Neck):
    """Prediction Neck for SSD

    Arguments:
        config
        in_channels (int):
    """
    def __init__(
        self,
        config,
        in_channels: int,
        extra_layers: Dict[str, List]
    ) -> None:
        super().__init__()
        self.config = config
        self.channels = []
        self.channels.append(in_channels)

        self.norm = L2Norm(512, 10)
        self.selected_layers = config.selected_layers
        self.extra_layers = nn.ModuleList()  # layers or extra_layers?
        self._in_channels = in_channels

        # TODO: rename variable
        for layer in extra_layers:
            self._add_extra_layer(layer)

    def _add_extra_layer(self, config, **kwargs):
        _layers = []
        for v in config:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]
                print(v, kwargs)
            if v == 'M':
                _layers.append(nn.MaxPool2d(**kwargs))
            else:
                if kwargs is None:
                    kwargs = {'kernel_size': 1}

                _layers += [
                    nn.Conv2d(
                        in_channels=self._in_channels, 
                        out_channels=v,
                        **kwargs),
                    nn.ReLU()]

                self._in_channels = v
            print(_layers)

        self.channels.append(self._in_channels)
        self.extra_layers.append(nn.Sequential(*_layers))

    def forward(self, inputs: List[Tensor]):
        outputs = []
        outputs.append(self.norm(inputs[-2]))

        output = inputs[-1]
        for layer in self.extra_layers:
            output = layer(output)
            outputs.append(output)

        for out in outputs:
            print(out.size())

        self.config.grid_sizes = [e.size(-1) for e in outputs]

        return outputs


def prior_cache(func):
    cache = defaultdict()

    @functools.wraps(func)
    def wrapper(*args):
        k, v = func(*args)
        if k not in cache:
            cache[k] = v
        return k, cache[k]
    return wrapper


class PriorBox:
    """Prior Box
    
    Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, config, aspect_ratios, step, min_sizes, max_sizes):
        self.max_size = config.max_size[0]
        self.aspect_ratios = aspect_ratios
        self.step = step
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes

        self.num_priors = len(config.aspect_ratios)
        self.variance = config.variance or [0.1]

        self.clip = config.clip
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
    
    @prior_cache
    def generate(self, h, w):
        size = (h, w)
        prior_boxes = []
        for i, j in itertools.product(range(h), repeat=2):
            f_k = self.max_size / self.step
            # unit center x,y
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio: 1
            # rel size: min_size
            s_k = self.min_sizes / self.max_size
            prior_boxes += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = math.sqrt(s_k * (self.max_sizes/self.max_size))
            prior_boxes += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            for ratio in self.aspect_ratios:
                prior_boxes += [cx, cy, s_k*math.sqrt(ratio), s_k/math.sqrt(ratio)]
                prior_boxes += [cx, cy, s_k/math.sqrt(ratio), s_k*math.sqrt(ratio)]
        # back to torch land
        prior_boxes = torch.tensor(prior_boxes).view(-1, 4)
        if self.clip:
            prior_boxes.clamp_(max=1, min=0)

        return size, prior_boxes


class SsdPredictHead(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        aspect_ratios: List[int],
        step: int,
        min_sizes: int,
        max_sizes: int
    ) -> None:
        super().__init__()
        self.config = config
        self.boxes = config.boxes
        
        self.prior_box = PriorBox(
            config, in_channels, aspect_ratios, step, min_sizes, max_sizes)

        self.bbox_layers = self._make_layer(in_channels, 4)
        self.conf_layers = self._make_layer(in_channels, config.num_classes + 1)

    def _add_predict_layer(self, in_channels, out_channels):
        _layer = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.boxes * out_channels, 
                kernel_size=3,
                padding=1)]
        
        return nn.Sequential(*_layer)

    def forward(self, inputs):
        h, w = inputs.size(2), inputs.size(3)
        bbox = self.bbox_layers(inputs)
        conf = self.bbox_layers(inputs)

        _, priors = self.prior_box.generate(h, w)

        return bbox, conf



# class SsdPredictHead(nn.Module):
#     def __init__(self, config, extra_channels: List[int]) -> None:
#         super().__init__()
#         self.config = config

#         self.num_classes = config.num_classes + 1
#         self.boxes = config.boxes
#         self.extra_channels = extra_channels

#         self.bbox_layers = nn.ModuleList()
#         self.conf_layers = nn.ModuleList()

#         self.prior_box = PriorBox(config, extra_channels)

#         self.bbox_layers = self._make_layer(4)
#         self.conf_layers = self._make_layer(config.num_classes + 1)
#         # self._make_layer()

#     # TODO: _make_layer or _add_predict_layer?
#     def _make_layer(self, out_channels):
#         _layers = []
#         for i in range(len(self.extra_channels)):
#             _layers += [
#                 nn.Conv2d(
#                     in_channels=self.extra_channels[i],
#                     out_channels=self.boxes[i] * out_channels, 
#                     kernel_size=3,
#                     padding=1)]
        
#         return nn.Sequential(*_layers)


#     def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
#         boxes = []
#         scores = []
#         for bbox_layer, output in zip(self.bbox_layers, inputs):
#             output = bbox_layer(output)
#             boxes.append(output)

#         for conf_layer, output in zip(self.conf_layers, inputs):
#             output = conf_layer(output)
#             scores.append(output)

#         priors = self.prior_box.generate()  # priors or prior_boxes?
#         # print(priors)
#         print(priors.size())
#         # print(self.bbox_layers[0].size())
#         boxes = torch.cat([box.view(box.size(0), -1) for box in boxes], 1)
#         scores = torch.cat([score.view(score.size(0), -1) for score in scores], 1)
        
        
#         # print(boxes.size())
#         # print(scores.size())
#         boxes = boxes.view(boxes.size(0), -1, 4)
#         scores = scores.view(scores.size(0), -1, self.num_classes)
#         # print(boxes.size(), scores.size())

#         preds = {
#             'boxes': boxes,
#             'scores': scores,}
#             # 'priors': priors}

#         return preds


class SsdPretrained(Model):
    config_class = SsdConfig
    base_model_prefix = 'ssd'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = SsdModel(config)

        return model


class SsdModel(SsdPretrained):
    """
     ██████╗  ██████╗ ███████╗ 
    ██╔════╝ ██╔════╝ ██╔═══██╗
    ╚██████╗ ╚██████╗ ██║   ██║
     ╚════██╗ ╚════██╗██║   ██║
     ██████╔╝ ██████╔╝███████╔╝
     ╚═════╝  ╚═════╝ ╚══════╝

    """
    def __init__(
        self,
        config,
        backbone=None,
        neck=None,
        head=None,
        **kwargs
    ) -> None:
        super().__init__(config)
        self.config = config
        self.backbone = vgg16()
        print(config.selected_layers)
        print(self.backbone.channels)
        self.neck = SsdPredictNeck(config, self.backbone.channels[-1], STRUCTURES['ssd300'])
        print('neck', self.neck.channels)

        self.head_layers = nn.ModuleList()
        for i, v in self.neck.channels:
            self.head = SsdPredictHead(
                config, v, config.aspect_ratios[i], config.steps[i], config.min_sizes[i], config.max_sizes[i])

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        print(len(outputs))
        outputs = self.neck(outputs)
        print(len(outputs))
        print(self.config.grid_sizes)
        outputs = self.head(outputs)
        print(len(outputs))


EXTRA_LAYER_STRUCTURES = {
    'ssd300': [
        [(256, {'kernel_size': 1}), (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3})]],
    'ssd512': [
        [(256, {'kernel_size': 1}), (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
        [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride': 2, 'padding':  1})],
        [(128, {'kernel_size': 1})]]
}
