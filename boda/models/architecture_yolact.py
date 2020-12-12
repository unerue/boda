import os
import math
import itertools
import functools
from collections import defaultdict
from collections import OrderedDict
from typing import Tuple, List, Dict, Any, Callable, TypeVar

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import Neck, Head, PretrainedModel
from .configuration_yolact import YolactConfig
from .backbone_resnet import resnet101
# from ..architecture_base import Register


class YolactPredictNeck(Neck):
    """Prediction Neack for YOLACT

    Arguments:
        in_channels ():
    """
    def __init__(self, config, in_channels: List[int]) -> None:
        super().__init__()
        self.config = config
        _selected_layers = list(range(len(config.selected_layers) + config.num_downsamples))
        self.channels = [config.fpn_out_channels] * len(_selected_layers)

        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(
                _in_channels,
                config.fpn_out_channels,
                kernel_size=1) for _in_channels in reversed(in_channels)])

        self.predict_layers = nn.ModuleList([
            nn.Conv2d(
                config.fpn_out_channels,
                config.fpn_out_channels,
                kernel_size=3,
                padding=config.padding) for _ in in_channels])

        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(
                config.fpn_out_channels,
                config.fpn_out_channels,
                kernel_size=3,
                stride=2,
                padding=1) for _ in range(config.num_downsamples)])

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        x = torch.zeros(1, device=self.config.device)
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

    @functools.wraps(func)
    def wrapper(*args):
        k, v = func(*args)
        if k not in cache:
            cache[k] = v
        return k, cache[k]
    return wrapper


class PriorBox:
    def __init__(self, config, aspect_ratios: List[int], scales: List[float]) -> None:
        self.config = config
        self.aspect_ratios = aspect_ratios
        self.scales = scales

    @prior_cache
    def generate(self, h, w):
        size = (h, w)
        prior_boxes = []
        for j, i in itertools.product(range(h), range(w)):
            x = (i + 0.5) / w
            y = (j + 0.5) / h

            for ratios in self.aspect_ratios:
                for scale in self.scales:
                    for ratio in ratios:
                        if not self.config.preapply_sqrt:
                            ratio = math.sqrt(ratio)

                        if self.config.use_pixel_scales:
                            w = scale * ratio / self.config.max_size[1]
                            h = scale / ratio / self.config.max_size[0]
                        else:
                            w = scale * ratio / h
                            h = scale / ratio / w

                        if self.config.use_square_anchors:
                            h = w

                        prior_boxes += [x, y, w, h]

        priors = torch.Tensor(prior_boxes).view(-1, 4)
        priors.requires_grad = False

        return size, priors


class ProtoNet(nn.Sequential):
    """ProtoNet of YOLACT

    Arguments:
        config ()
        in_channels ()
        layers ()
        include_last_relu ()        
    """
    def __init__(
        self,
        config,
        in_channels: int,
        layers: List,
        include_last_relu: bool = True) -> None:
        self.config = config
        self.channels = []

        mask_layers = OrderedDict()
        for i, v in enumerate(layers):
            if isinstance(v[0], int):
                mask_layers[f'{i}'] = nn.Conv2d(
                    in_channels, v[0], kernel_size=v[1], **v[2])
                self.channels.append(v[0])

            elif v[0] is None:
                mask_layers[f'{i}'] = nn.Upsample(
                    scale_factor=-v[1], mode='bilinear', align_corners=False, **v[2])

        if include_last_relu:
            mask_layers[f'relu{len(mask_layers)+1}'] = nn.ReLU()

        super().__init__(mask_layers)


class YolactPredictHead(Head):
    """Prediction Head for YOLACT

    Arguments:
        config
        in_channles
        out_channels
        aspect_ratio
        scales
        parent
        index
    """
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        aspect_ratios: List[int],
        scales: List[float],
        parent,
        index: int) -> None:
        super().__init__()
        self.config = config
        self.prior_box = PriorBox(config, aspect_ratios, scales)
        self.config.mask_dim = config.mask_dim
        self.num_priors = sum(len(x)*len(scales) for x in aspect_ratios)
        self.parent = [parent]

        if parent is None:
            if config.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature = ProtoNet(config, in_channels, config.extra_head_net)
                out_channels = self.upfeature.channels[-1]

            self.bbox_layers = self._add_predict_layer(
                config.num_extra_bbox_layers,
                out_channels,
                self.num_priors * 4)

            self.conf_layers = self._add_predict_layer(
                config.num_extra_conf_layers,
                out_channels,
                self.num_priors * config.num_classes)

            self.mask_layers = self._add_predict_layer(
                config.num_extra_mask_layers,
                out_channels,
                self.num_priors * self.config.mask_dim)

    # TODO: _add_predict_layer or _make_layer?
    def _add_predict_layer(
        self,
        num_extra_layers: int,
        in_channels: int,
        out_channels: int) -> nn.Sequential:
        _predict_layers = []
        if num_extra_layers > 0:
            for _ in range(num_extra_layers):
                _predict_layers += [
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=1),
                    nn.ReLU()]

        _predict_layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

        return nn.Sequential(*_predict_layers)

    # def forward(self, inputs: Tensor) -> Dict(str, Tensor):
    def forward(self, inputs: Tensor):
        pred = self if self.parent[0] is None else self.parent[0]

        h, w = inputs.size(2), inputs.size(3)

        inputs = pred.upfeature(inputs)

        bbox = pred.bbox_layers(inputs)
        conf = pred.conf_layers(inputs)
        mask = pred.mask_layers(inputs)

        bbox = bbox.permute(0, 2, 3, 1).contiguous().view(inputs.size(0), -1, 4)
        conf = conf.permute(0, 2, 3, 1).contiguous().view(inputs.size(0), -1, self.config.num_classes)

        _, priors = self.prior_box.generate(h, w)

        return bbox, conf, mask, priors


class YolactPretrained(Model):
    config_class = YolactConfig
    base_model_prefix = 'yolact'

    # @classmethod
    # def from_pretrained(cls, model_name_or_path):
    #     print('Loading model!')
    #     config, model_kwargs = cls.config_class.from_pretrained(model_name_or_path)
    #     config = YolactConfig()
    #     model = YolactModel(config)
    #     model.state_dict(torch.load('yolact.pth'))

    #     return model


class YolactModel(YolactPretrained):
    """
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝

    Arguments:
        config ()
        backbone ()
        neck ()
        head ()

    """
    model_name = 'yolact'

    def __init__(
        self,
        config,
        backbone=None,
        neck=None,
        head=None) -> None:
        super().__init__(config)
        self.config = config

        if backbone is None:
            self.backbone = resnet101()
            num_layers = max(config.selected_layers) + 1
            while len(self.backbone.layers) < num_layers:
                self.backbone.add_layer()

        self.config.mask_dim = config.mask_size**2

        in_channels = self.backbone.channels[config.proto_src]
        in_channels += config.num_grids

        if neck is None:
            self.neck = YolactPredictNeck(
                config, [self.backbone.channels[i] for i in config.selected_layers])

        in_channels = config.fpn_out_channels
        in_channels += config.num_grids

        self.proto_net = ProtoNet(
            config, in_channels, config.proto_net, include_last_relu=False)

        self.config.mask_dim = self.proto_net.channels[-1]

        self.head_layers = nn.ModuleList()
        self.config.num_heads = len(config.selected_layers)
        for i, j in enumerate(config.selected_layers):
            parent = None
            if i > 0:
                parent = self.head_layers[0]

            head_layer = YolactPredictHead(
                config,
                self.neck.channels[j],
                self.neck.channels[j],
                aspect_ratios=config.aspect_ratios[i],
                scales=config.scales[i],
                parent=parent,
                index=i)
            # print(config.aspect_ratios[i], config.scales[i])
            self.head_layers.append(head_layer)

        self.semantic_layer = nn.Conv2d(self.neck.channels[0], config.num_classes-1, kernel_size=1)

    def forward(self, inputs):
        inputs = self.check_inputs(inputs)
        self.config.device = inputs.device
        print(self.config.device)

        self.config.size = (inputs.size(2), inputs.size(3))

        outputs = self.backbone(inputs)
        print(self.backbone.channels)

        outputs = [outputs[i] for i in self.config.selected_layers]
        outputs = self.neck(outputs)
        print(len(outputs))
        for o in outputs:
            print(o.size())

        proto_input = outputs[0]
        # print(proto_input.size())
        segout = self.semantic_layer(outputs[0])
        proto_output = self.proto_net(proto_input)
        for i, layer in zip(self.config.selected_layers, self.head_layers):
            print(i, layer)
            print(outputs[i].size())
            output = layer(outputs[i])

        return outputs
