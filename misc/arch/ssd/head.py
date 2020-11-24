import os
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import itertools 


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