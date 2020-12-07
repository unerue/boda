import os
import math
from typing import List
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import itertools 
from .architecture_base import BaseModel
from .backbone_vgg import vgg16


class AnchorGenerator(nn.Module):
    def __init__(self):
        super().__init__()


class RegionProposalHead:
    pass



class MultiScaleRoIAlign:
    pass

class TwoMLPHead:
    pass

class FastRcnnHead:
    pass

class RoIHeads:
    pass

class PredictHead(nn.Module):
    """RPN head """
    def __init__(self, config, in_channels):
        super().__init__()
        self.config = config

        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d()
        self.bbox_layers = nn.Conv2d()

    def forward(self, inputs):
        pass

class RegionProposalNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.anchor_generator = AnchorGenerator()
        self.head = None
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def forward(self, inputs):
        pass

    

class FasterRcnnModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads


        self.rpn_anchor_generator = AnchorGenerator(
            self.anchor_sizes, self.aspect_ratios)

        self.rpn_head = 

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        proposals = self.rpn(inputs, features)
    