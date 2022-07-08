# import os
# import math
# import functools
# from collections import OrderedDict, defaultdict
# from typing import List, Dict, Union
# import torch
# from torch import nn, Tensor
# import torch.nn.functional as F

# import itertools 
# from ...base_architecture import Neck, Head, Model
# from .configuration_ssd import SsdConfig
# from ..feature_extractor import vgg16


# class L2Norm(nn.Module):
#     """ L2 Normalization

#     Args:
#         in_channels (:obj:`int`):
#         gamma ():
#         eps ():

#     Adpated from:
#     """
#     def __init__(
#         self,
#         in_channels: int = 512,
#         gamma: int = 10,
#         eps: float = 1e-10,
#     ) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         self.gamma = gamma
#         self.eps = eps
#         self.weight = nn.Parameter(torch.Tensor(in_channels))
#         self.init_weights()

#     def init_weights(self):
#         torch.nn.init.constant_(self.weight, self.gamma)

#     def forward(self, inputs):
#         norm_inputs = inputs.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
#         inputs = torch.div(inputs, norm_inputs)  # x /= norm_x
#         outputs = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(inputs) * inputs
#         return outputs


# class SsdPredictNeck(Neck):
#     """Prediction Neck for SSD

#     Args:
#         config
#         in_channels (:obj:`int`):
#     """
#     def __init__(
#         self,
#         config,
#         in_channels: int,
#         structure: Dict[str, List]
#     ) -> None:
#         super().__init__()
#         self.config = config
#         self.channels = []
#         self.channels.append(in_channels)
#         self.extra_layers = nn.ModuleList()

#         self.norm = L2Norm(512, 10)
#         self.selected_layers = config.selected_layers
#         self.layers = nn.ModuleList()
#         self._in_channels = in_channels

#         # TODO: rename variable
#         for layer in structure:
#             self._make_layer(layer)

#     def _make_layer(self, config, **kwargs):
#         # _layers = []
#         _layers = OrderedDict()
#         i = 0
#         for v in config:
#             kwargs = None
#             if isinstance(v, tuple):
#                 kwargs = v[1]
#                 v = v[0]

#             if v == 'M':
#                 # _layers.append(nn.MaxPool2d(**kwargs))
#                 _layers.update({
#                     f'maxpool{i}': nn.MaxPool2d(**kwargs)
#                 })
#             else:
#                 if kwargs is None:
#                     kwargs = {'kernel_size': 1}

#                 _layers.update({
#                     (f'{i}', nn.Conv2d(
#                         in_channels=self._in_channels,
#                         out_channels=v,
#                         **kwargs)),
#                     (f'relu{i}', nn.ReLU())
#                 })

#                 self._in_channels = v
#             i += 1

#         self.channels.append(self._in_channels)
#         self.layers.append(nn.Sequential(_layers))

#     def forward(self, inputs: List[Tensor]):
#         outputs = []
#         outputs.append(self.norm(inputs[-2]))

#         output = inputs[-1]
#         for layer in self.layers:
#             output = layer(output)
#             outputs.append(output)

#         self.config.grid_sizes = [e.size(-1) for e in outputs]

#         return outputs


# def prior_cache(func):
#     cache = defaultdict()

#     @functools.wraps(func)
#     def wrapper(*args):
#         k, v = func(*args)
#         if k not in cache:
#             cache[k] = v
#         return k, cache[k]
#     return wrapper


# class PriorBox:
#     """Prior Box

#     Compute priorbox coordinates in center-offset form for each source
#     feature map.
#     """
#     def __init__(self, config, aspect_ratios, step, min_sizes, max_sizes):
#         self.max_size = config.max_size[0]
#         self.aspect_ratios = aspect_ratios
#         self.step = step
#         self.min_sizes = min_sizes
#         self.max_sizes = max_sizes

#         self.num_priors = len(config.aspect_ratios)
#         self.variance = config.variance or [0.1]

#         self.clip = config.clip
#         for v in self.variance:
#             if v <= 0:
#                 raise ValueError('Variances must be greater than 0')

#     @prior_cache
#     def generate(self, h, w):
#         size = (h, w)
#         prior_boxes = []
#         for i, j in itertools.product(range(h), repeat=2):
#             f_k = self.max_size / self.step
#             # unit center x,y
#             cx = (j + 0.5) / f_k
#             cy = (i + 0.5) / f_k

#             # aspect_ratio: 1
#             # rel size: min_size
#             s_k = self.min_sizes / self.max_size
#             prior_boxes += [cx, cy, s_k, s_k]

#             # aspect_ratio: 1
#             # rel size: sqrt(s_k * s_(k+1))
#             s_k_prime = math.sqrt(s_k * (self.max_sizes/self.max_size))
#             prior_boxes += [cx, cy, s_k_prime, s_k_prime]

#             # rest of aspect ratios
#             for ratio in self.aspect_ratios:
#                 prior_boxes += [cx, cy, s_k*math.sqrt(ratio), s_k/math.sqrt(ratio)]
#                 prior_boxes += [cx, cy, s_k/math.sqrt(ratio), s_k*math.sqrt(ratio)]
#         # back to torch land
#         prior_boxes = torch.tensor(prior_boxes).view(-1, 4)
#         prior_boxes.require_grad = False
#         if self.clip:
#             prior_boxes.clamp_(max=1, min=0)

#         return size, prior_boxes


# class BoxBranch(nn.Conv2d):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             padding=1
#         )


# class ScoreBranch(nn.Conv2d):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             padding=1
#         )


# class SsdPredictHeads(nn.ModuleList):
#     def __init__(modules):
#         super().__init__(modules)


# class SsdPredictHead(nn.Module):
#     def __init__(
#         self,
#         config,
#         in_channels: int,
#         boxes: int,
#         aspect_ratios: List[int],
#         step: int,
#         min_sizes: int,
#         max_sizes: int
#     ) -> None:
#         super().__init__()
#         self.config = config
#         self.num_classes = config.num_classes + 1
#         self.boxes = boxes

#         self.prior_box = PriorBox(
#             config, aspect_ratios, step, min_sizes, max_sizes)

#         # self.box_layer = nn.Conv2d(
#         #     in_channels=in_channels,
#         #     out_channels=self.boxes * 4,
#         #     kernel_size=3,
#         #     padding=1
#         # )

#         # self.score_layer = nn.Conv2d(
#         #     in_channels=in_channels,
#         #     out_channels=self.boxes * self.num_classes,
#         #     kernel_size=3,
#         #     padding=1
#         # )
#         self.box_layer = BoxBranch(in_channels, self.boxes*4)
#         self.score_layer = ScoreBranch(in_channels, self.boxes*self.num_classes)

#     def forward(self, inputs):
#         h, w = inputs.size(2), inputs.size(3)
#         boxes = self.box_layer(inputs)
#         scores = self.score_layer(inputs)

#         _, prior_boxes = self.prior_box.generate(h, w)

#         boxes = boxes.view(boxes.size(0), -1, 4)
#         scores = scores.view(scores.size(0), -1, self.num_classes)

#         return_dict = {
#             'boxes': boxes,
#             'scores': scores,
#             'prior_boxes': prior_boxes
#         }

#         return return_dict


# class SsdPretrained(Model):
#     config_class = SsdConfig
#     base_model_prefix = 'ssd'

#     @classmethod
#     def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
#         config = cls.config_class.from_pretrained(name_or_path)
#         model = SsdModel(config)

#         return model


# class SsdModel(SsdPretrained):
#     """
#     """
#     structures = {
#         'ssd300': [
#             [('M', {'kernel_size': 3, 'stride': 1, 'padding': 1}),
#              (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
#              (1024, {'kernel_size': 1})],
#             [(256, {'kernel_size': 1}), (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
#             [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
#             [(128, {'kernel_size': 1}), (256, {'kernel_size': 3})],
#             [(128, {'kernel_size': 1}), (256, {'kernel_size': 3})]],
#         'ssd512': [
#             [(256, {'kernel_size': 1}), (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
#             [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
#             [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})],
#             [(128, {'kernel_size': 1}), (256, {'kernel_size': 3, 'stride': 2, 'padding':  1})],
#             [(128, {'kernel_size': 1})]]
#     }

#     def __init__(
#         self,
#         config,
#         backbone=None,
#         neck=None,
#         head=None,
#         **kwargs
#     ) -> None:
#         super().__init__(config)
#         self.config = config
#         self.backbone = vgg16()
#         self.neck = SsdPredictNeck(config, self.backbone.channels[-1], self.structures['ssd300'])

#         self.heads = SsdPredictHeads()
#         for i, in_channels in enumerate(self.neck.channels):
#             head = SsdPredictHead(
#                 config,
#                 in_channels, config.boxes[i], config.aspect_ratios[i], config.steps[i], config.min_sizes[i], config.max_sizes[i])
#             self.heads.append(head)

#     def forward(self, inputs):
#         inputs = self.check_inputs(inputs)
#         outputs = self.backbone(inputs)
#         outputs = self.neck(outputs)

#         preds = defaultdict(list)
#         for i, layer in enumerate(self.heads):
#             output = layer(outputs[i])

#             for k, v in output.items():
#                 preds[k].append(v)

#         for k, v in preds.items():
#             preds[k] = torch.cat(v, dim=-2)

#         if self.training:
#             return preds
#         else:
#             for k, v in preds.items():
#                 print(k, v.size())

#             preds['scores'] = F.softmax(preds['scores'], dim=-1)
#             return preds

#     def load_weights(self, path):
#         state_dict = torch.load(path)
#         _norm = state_dict.pop('L2Norm.weight')
#         keys = list(self.state_dict().keys())
#         keys.remove('neck.norm.weight')
#         # state_dict['norm.weight'] = state_dict
#         for key1, key2 in zip(list(state_dict.keys()), keys):
#             state_dict[key2] = state_dict.pop(key1)
#             if key1.startswith('loc.0'):
#                 break

#         state_dict['neck.norm.weight'] = _norm

#         for key in list(state_dict.keys()):
#             p = key.split('.')
#             # if p[0] == 'vgg':
#             #     new_key = f'backbone.conv.{p[2]}'
#             #     state_dict[new_key] = state_dict.pop(key)
#             # elif key.startswith('vgg.3'):
#             #     new_key = f'neck'
#             #     state_dict[new_key] = state_dict.pop(key)
#             # elif p[0] == 'extras':
#             #     new_key = f'neck'
#             #     state_dict[new_key] = state_dict.pop(key)
#             # elif p[0] == 'L2Norm':
#             #     new_key = f'norm.{p[1]}'
#             #     state_dict[new_key] = state_dict.pop(key)
#             if p[0] == 'loc':
#                 new_key = f'heads.{p[1]}.box_layer.{p[2]}'
#                 state_dict[new_key] = state_dict.pop(key)
#             elif p[0] == 'conf':
#                 new_key = f'heads.{p[1]}.score_layer.{p[2]}'
#                 state_dict[new_key] = state_dict.pop(key)

#         self.load_state_dict(state_dict)
