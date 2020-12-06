import sys
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
from torch import nn, Tensor
from torch.functional import align_tensors

from .architecture_base import BaseModel
from .backbone_darknet import darknet21


class RetinaNetPredictNeck(nn.Module):
    """Prediction Head for RetinaNet"""
    def __init__(self, config, in_channels) -> None:
        super().__init__()
        self.config = config
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(
                x, 
                self.config.num_features,
                kernel_size=1) for x in reversed(in_channels)])

        self.upsample_layers = nn.ModuleList([
             nn.Upsample(
                 scale_factor=2, 
                 mode='nearest') for _ in range(self.config.num_upsamples)])

    def forward(self, inputs):
        outputs = []

        j = len(inputs)
        for lateral_layer in self.lateral_layers:
            j -= 1
            if j < len(inputs) - 1:
                _, _, h, w = inputs[j].size()
                x = F.interpolate(
                    x, size=(h, w), mode='nearest', align_corner=False)

        return


class RetinaNetPredictHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, inputs):
        return


class RetinaNetModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, inputs):
        return