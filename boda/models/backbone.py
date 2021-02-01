from typing import List

import torch
from torch import nn, Tensor

from .backbone_resnet import resnet18, resnet34, resnet50, resnet101
from .backbone_vggnet import vgg16


class Backbone:
    backbones = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'vgg16': vgg16,
    }

    def __init__(self) -> None:
        ...

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        selected_layers = kwargs.get('selected_layers', [1, 2, 3])
        backbone = self.backbones[name]
        num_layers = max(selected_layers) + 1
        while len(backbone.layers) < num_layers:
            backbone.add_layer()

        return backbone

    @property
    def backbone_list(self):
        return self.backbones


if __name__ == '__main__':
    backbone = Backbone.from_pretrained('resnet50')
    print(type(backbone))
    print(backbone.backbone_list)
