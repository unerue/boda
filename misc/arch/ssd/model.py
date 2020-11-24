import os
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import itertools 


from ..backbone import vgg


backbone_layers = [
    [64, 64],
    ['M', 128, 128],
    ['M', 256, 256, 256],
    [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
    ['M', 512, 512, 512],
    # [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
    #  (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
    #  (1024, {'kernel_size': 1})]
]


extra_layers = [
    [('M', {'kernel_size': 3, 'stride':  1, 'padding':  1}),
     (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}), 
     (1024, {'kernel_size': 1})], 
    [(256, {'kernel_size': 1}), 
     (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
    [(128, {'kernel_size': 1}), 
     (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
    [(128, {'kernel_size': 1}), 
     (256, {'kernel_size': 3})], 
    [(128, {'kernel_size': 1}), 
     (256, {'kernel_size': 3})]]


# class L2Norm(nn.Module):
#     def __init__(self, n_channels, scale):
#         super().__init__()
#         self.n_channels = n_channels
#         self.gamma = scale or None
#         self.eps = 1e-10
#         self.weight = nn.Parameter(torch.Tensor(self.n_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.constant_(self.weight, val=self.gamma)

#     def forward(self, x):
#         norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
#         #x /= norm
#         x = torch.div(x, norm)
#         out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x

#         return out

config = {
    'selected_layers': [4, 6, 6, 6, 4, 4],
    'num_classes': 20,
    # Prior box configs
    'min_dim': 300,
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],

    'feature_maps': [38, 19, 10, 5, 3, 1], # 'feature_map_sizes'

    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'steps': [8, 16, 32, 64, 100, 300],
    'clip': True,
}


class PriorBox:
    """Prior Box
    Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, config):
        super().__init__()
        self.image_size = config['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(config['aspect_ratios'])
        self.aspect_ratios = config['aspect_ratios']

        self.variance = config['variance'] or [0.1]
        self.feature_maps = config['feature_maps']
        self.min_sizes = config['min_sizes']
        self.max_sizes = config['max_sizes']
        self.steps = config['steps']
        
        self.clip = config['clip']
        self.version = config['name']

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = math.sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                    mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
        # back to torch land
        output = torch.FloatTensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
            
        return output







class SsdModel(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super().__init__()
        self.prediction_head = SsdPredictionHead()
        self.phase = phase
        self.num_classes = num_classes

    def forward(self, x):
        pass
        return output











