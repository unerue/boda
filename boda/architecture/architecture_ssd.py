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




class SsdPredictionHead(nn.Module):
    """SSD Prediction Neck and Head
    """
    def __init__(self, config = None):
        super().__init__()

        self.num_classes = 20 + 1

        # self.backbone_layers = config.backbone_layers
        self.backbone = vgg(backbone_layers)

        self.extra_channels = []
        self.extra_channels.append(self.backbone.channels[-1])
        self.extra_layers = nn.ModuleList()

        self.in_channels = self.backbone.channels[-1]
        
        for cfg in extra_layers:
            self._add_extra_layers(cfg)

        print(self.extra_channels)
        

        self.selected_layers = [4, 6]

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        
        self._multibox()

        self.l2_norm = None
        self.prior_box = None
        print(self.loc_layers)
        print(self.conf_layers)
        print(len(self.loc_layers))
        print(len(self.conf_layers))

        # self._multibox(config.selected_layers)
        
    def _add_extra_layers(self, config):
        # Extra layers added to VGG for feature scaling
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
                        in_channels=self.in_channels, 
                        out_channels=v,
                        **kwargs),
                    nn.ReLU()]

                self.in_channels = v

        self.extra_channels.append(v)
        self.extra_layers.append(nn.Sequential(*layers))

    def _multibox(self):
        """config -? VGG selected layers"""
        config = [4, 6, 6, 6, 4, 4]
        for i in range(len(self.extra_channels)):
            self.loc_layers += [
                nn.Conv2d(
                    in_channels=self.extra_channels[i],
                    out_channels=config[i] * 4, 
                    kernel_size=3, 
                    padding=1)]

            self.conf_layers += [
                nn.Conv2d(
                    in_channels=self.extra_channels[i],
                    out_channels=config[i] * self.num_classes, 
                    kernel_size=3, 
                    padding=1)]


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        outputs = []
        x = self.backbone(x)
        # TODO:
        # 마지막 백본 레이어에 L2Norm 추가
        # PriorBox 추가
        x = x[-1]

        outputs.append(x)
        for layer in self.extra_layers:
            x = layer(x)
            outputs.append(x)

        loc_layers = []
        conf_layers = []
        for output, loc_layer, conf_layer in zip(outputs, self.loc_layers, self.conf_layers):
            loc_layers.append(loc_layer(output).permute(0, 2, 3, 1).contiguous())
            conf_layers.append(conf_layer(output).permute(0, 2, 3, 1).contiguous())

        boxes = torch.cat([out.view(out.size(0), -1) for out in loc_layers], dim=1)
        scores = torch.cat([out.view(out.size(0), -1) for out in conf_layers], dim=1)

        outputs = {
            'boxes': boxes.view(boxes.size(0), -1, 4),
            'scores': scores.view(scores.size(0), -1, self.num_classes),
            # 'priors': self.priors
        }
        print(outputs['boxes'].size())
        print(outputs['scores'].size())
        return outputs


class SsdModel(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super().__init__()
        self.prediction_head = SsdPredictionHead()
        self.phase = phase
        self.num_classes = num_classes

    def forward(self, x):
        pass
        return output











