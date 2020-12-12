import os
import math
from typing import List
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import itertools 
from .architecture_base import BaseModel
from ..base import Neck, Head, Model
from .backbone_vgg import vgg16


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


class SsdPredictNeck(nn.Module):
    def __init__(self, config, in_channels: int):
        super().__init__()
        self.config = config
        self.out_channels = []
        self.norm = L2Norm(512, 10)

        self.selected_layers = config.selected_layers
        self.out_channels.append(in_channels)
        self.layers = nn.ModuleList()  # layers or extra_layers?
        self._in_channels = in_channels

        for cfg in extra_layers:
            self._add_extra_layer(cfg)

    def _add_extra_layer(self, config, **kwargs):
        layers = []
        for v in config:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]
            print(self.in_channels, v, kwargs)

            if v == 'M':
                layers.append(nn.MaxPool2d(**kwargs))
            else:
                if kwargs is None:
                    kwargs = {'kernel_size': 1}

                layers += [
                    nn.Conv2d(
                        in_channels=self._in_channels, 
                        out_channels=v,
                        **kwargs),
                    nn.ReLU()]

                self._in_channels = v

        self.out_channels.append(v)
        self.layers.append(nn.Sequential(*layers))

    def forward(self, inputs: List[Tensor]):
        outputs = []
        outputs.append(self.norm(inputs[-2]))

        output = inputs[-1]
        for layer in self.layers:
            output = layer(output)
            outputs.append(output)

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
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, config, feature_maps=None):
        self.max_size = config.max_size[0]
        self.num_priors = len(config.aspect_ratios)
        self.variance = config.variance or [0.1]
        self.feature_maps = config.grid_sizes
        self.min_sizes = config.min_sizes
        self.max_sizes = config.max_sizes
        self.steps = config.steps
        self.aspect_ratios = config.aspect_ratios
        self.clip = config.clip
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
    
    @prior_cache
    def generate(self):
        prior_boxes = []
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.max_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.max_size
                prior_boxes += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = math.sqrt(s_k * (self.max_sizes[k]/self.max_size))
                prior_boxes += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ratio in self.config.aspect_ratios[k]:
                    prior_boxes += [cx, cy, s_k*math.sqrt(ratio), s_k/math.sqrt(ratio)]
                    prior_boxes += [cx, cy, s_k/math.sqrt(ratio), s_k*math.sqrt(ratio)]
        # back to torch land
        output = torch.tensor(prior_boxes).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        
        return output


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, inputs):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        #x /= norm
        inputs = torch.div(inputs, norm)
        outputs = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(inputs) * inputs
        return outputs


class SsdPredictHead(nn.Module):
    def __init__(self, config, extra_channels):
        super().__init__()
        self.config = config
        self.out_channels = []

        self.num_classes = config.num_classes + 1
        self.boxes = config.boxes
        self.extra_channels = extra_channels
        self.bbox_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        self.prior_box = PriorBox(config)

        self.bbox_layers = self._make_layer(4)
        self.conf_layers = self._make_layer(config.num_classes + 1)
        # self._make_layer()

    # TODO: _make_layer or _add_predict_layer?
    def _make_layer(self, out_channels):
        _layers = []
        for i in range(len(self.extra_channels)):
            _layers += [
                nn.Conv2d(
                    in_channels=self.extra_channels[i],
                    out_channels=self.boxes[i] * out_channels, 
                    kernel_size=3,
                    padding=1)]
        
        return nn.Sequential(*_layers)

    # def _make_layer(self):
    #     layers = []
    #     for i in range(len(self.extra_channels)):
    #         self.bbox_layers += [
    #             nn.Conv2d(
    #                 in_channels=self.extra_channels[i],
    #                 out_channels=self.boxes[i] * 4, 
    #                 kernel_size=3,
    #                 padding=1)]

    #         self.conf_layers += [
    #             nn.Conv2d(
    #                 in_channels=self.extra_channels[i],
    #                 out_channels=self.boxes[i] * self.num_classes, 
    #                 kernel_size=3,
    #                 padding=1)]

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        bbox = []
        conf = []
    
        for bbox_layer, output in zip(self.bbox_layers, inputs):
            output = bbox_layer(output)
            bbox.append(output)

        for conf_layer, output in zip(self.conf_layers, inputs):
            output = conf_layer(output)
            conf.append(output)

        priors = prior_box.generate()  # priors or prior_boxes?
        print(priors)
        print(priors.size())
        print(bbox_layers[0].size())
        bbox = torch.cat([o.view(o.size(0), -1) for o in bbox], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        
        print(bbox.size())
        print(conf.size())
        boxes = bbox.view(bbox.size(0), -1, 4)
        scores = conf.view(conf.size(0), -1, self.num_classes)
        print(bbox.size(), conf.size())

        preds = {
            'boxes': boxes,
            'scores': scores,
            'priors': priors}

        return preds


class SsdModel(BaseModel):
    """
     ██████╗  ██████╗ ███████╗ 
    ██╔════╝ ██╔════╝ ██╔═══██╗
    ╚██████╗ ╚██████╗ ██║   ██║
     ╚════██╗ ╚════██╗██║   ██║
     ██████╔╝ ██████╔╝███████╔╝
     ╚═════╝  ╚═════╝ ╚══════╝

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = vgg16()
        self.neck = SsdPredictNeck(config, self.backbone.channels[-1])
        self.head = SsdPredictHead(config, self.neck.out_channels)

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        print(len(outputs))
        outputs = self.neck(outputs)
        print(len(outputs))
        print(self.config.grid_sizes)
        outputs = self.head(outputs)
        print(len(outputs))

        # print(outputs[0][0])




# extra_layers = [
#     # [('M', {'kernel_size': 3, 'stride':  1, 'padding':  1}),
#     #  (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}), 
#     #  (1024, {'kernel_size': 1})], 
#     [(256, {'kernel_size': 1}), 
#      (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
#     [(128, {'kernel_size': 1}), 
#      (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
#     [(128, {'kernel_size': 1}), 
#      (256, {'kernel_size': 3})], 
#     [(128, {'kernel_size': 1}), 
#      (256, {'kernel_size': 3})]]

# layers512 = [
#     [(256, {'kernel_size': 1}), 
#      (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
#     [(128, {'kernel_size': 1}), 
#      (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
#     [(128, {'kernel_size': 1}), 
#      (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
#     [(128, {'kernel_size': 1}), 
#      (256, {'kernel_size': 3, 'stride': 2, 'padding':  1})], 
#     [(128, {'kernel_size': 1})]]






# class SsdPredictHead(nn.Module):
#     """SSD Prediction Neck and Head
#     """
#     def __init__(self, config, ):
#         super().__init__()
#         self.num_classes = 20 + 1

#         # self.backbone_layers = config.backbone_layers
#         self.backbone = vgg(backbone_layers)

#         self.extra_channels = []
#         self.extra_channels.append(self.backbone.channels[-1])
#         self.extra_layers = nn.ModuleList()

#         self.in_channels = self.backbone.channels[-1]
        
#         for cfg in extra_layers:
#             self._add_extra_layers(cfg)

#         print(self.extra_channels)
        

#         self.selected_layers = [4, 6]

#         self.loc_layers = nn.ModuleList()
#         self.conf_layers = nn.ModuleList()

        
#         self._multibox()

#         self.l2_norm = None
#         self.prior_box = None
#         print(self.loc_layers)
#         print(self.conf_layers)
#         print(len(self.loc_layers))
#         print(len(self.conf_layers))

#         # self._multibox(config.selected_layers)
        
#     def _add_extra_layers(self, config):
#         # Extra layers added to VGG for feature scaling
#         layers = []
#         for v in config:
#             kwargs = None
#             if isinstance(v, tuple):
#                 kwargs = v[1]
#                 v = v[0]
#             print(self.in_channels, v, kwargs)
#             if v == 'M':
#                 layers.append(nn.MaxPool2d(**kwargs))
#             else:
#                 if kwargs is None:
#                     kwargs = {'kernel_size': 1}

#                 layers += [
#                     nn.Conv2d(
#                         in_channels=self.in_channels, 
#                         out_channels=v,
#                         **kwargs),
#                     nn.ReLU()]

#                 self.in_channels = v

#         self.extra_channels.append(v)
#         self.extra_layers.append(nn.Sequential(*layers))

#     def _multibox(self):
#         """config -? VGG selected layers"""
#         config = [4, 6, 6, 6, 4, 4]
#         for i in range(len(self.extra_channels)):
#             self.loc_layers += [
#                 nn.Conv2d(
#                     in_channels=self.extra_channels[i],
#                     out_channels=config[i] * 4, 
#                     kernel_size=3, 
#                     padding=1)]

#             self.conf_layers += [
#                 nn.Conv2d(
#                     in_channels=self.extra_channels[i],
#                     out_channels=config[i] * self.num_classes, 
#                     kernel_size=3, 
#                     padding=1)]


#     def forward(self, x):
#         """Applies network layers and ops on input image(s) x.
#         Args:
#             x: input image or batch of images. Shape: [batch,3,300,300].
#         Return:
#             Depending on phase:
#             test:
#                 Variable(tensor) of output class label predictions,
#                 confidence score, and corresponding location predictions for
#                 each object detected. Shape: [batch,topk,7]
#             train:
#                 list of concat outputs from:
#                     1: confidence layers, Shape: [batch*num_priors,num_classes]
#                     2: localization layers, Shape: [batch,num_priors*4]
#                     3: priorbox layers, Shape: [2,num_priors*4]
#         """
#         outputs = []
#         x = self.backbone(x)
#         # TODO:
#         # 마지막 백본 레이어에 L2Norm 추가
#         # PriorBox 추가
#         x = x[-1]

#         outputs.append(x)
#         for layer in self.extra_layers:
#             x = layer(x)
#             outputs.append(x)

#         loc_layers = []
#         conf_layers = []
#         for output, loc_layer, conf_layer in zip(outputs, self.loc_layers, self.conf_layers):
#             loc_layers.append(loc_layer(output).permute(0, 2, 3, 1).contiguous())
#             conf_layers.append(conf_layer(output).permute(0, 2, 3, 1).contiguous())

#         boxes = torch.cat([out.view(out.size(0), -1) for out in loc_layers], dim=1)
#         scores = torch.cat([out.view(out.size(0), -1) for out in conf_layers], dim=1)

#         outputs = {
#             'boxes': boxes.view(boxes.size(0), -1, 4),
#             'scores': scores.view(scores.size(0), -1, self.num_classes),
#             # 'priors': self.priors
#         }
#         print(outputs['boxes'].size())
#         print(outputs['scores'].size())
#         return outputs