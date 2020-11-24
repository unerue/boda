from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F


import torch
import torch.nn as nn
import os
import math
from collections import deque
from pathlib import Path
from layers.interpolate import InterpolateModule


class FPN(nn.Module):
    """Feature Pyramid Networks (FPN) for YOLACT
    https://arxiv.org/pdf/1612.03144.pdf
    """
    def __init__(self, in_channels: List = [512, 1024, 2048]):
        super().__init__()
        self.fpn_num_features = 256  # REPLACE CONFIG!
        self.fpn_pad = 1  # REPLACE CONFIG!
        # self.fpn_num_downsample = 2

        self.conv1 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        
        # Lateral layers
        lateral_channels = [512, 1024, 2048]
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=self.fpn_num_features, 
                kernel_size=1, 
                stride=1, 
                padding=0) for in_channels in reversed(lateral_channels)])

        # Top-down layers
        fpn_num_downsample = 2
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(
                self.fpn_num_features, 
                self.fpn_num_features,
                kernel_size=3, 
                stride=1,
                padding=1) for _ in range(fpn_num_downsample)])

    def forward(self, inputs: List[Tensor]):
        """
        backbone_outs = [[n, 512, 69, 69], [n, 1024, 35, 35], [n, 2048, 18, 18]]
        In class Yolact's train(), remove C2 from bakebone_outs. So FPN gets three feature outs.
        """
        outputs = []
        x = torch.zeros(1, device=inputs[0].device)
        for _ in range(len(inputs)):
            outputs.append(x)

        j = len(inputs)
        for lateral_layer in self.lateral_layers:
            j -= 1
            if j < len(inputs) - 1:
                _, _, h, w = inputs[j].size()
                x = F.interpolate(x, size=(h, w), mode='bilinear')
            
            x = x + lateral_layer(inputs[j])
            outputs[j] = x

        j = len(inputs)
        for pred_layer in self.pred_layers:
            j -= 1
            outputs[j] = F.relu(pred_layer(outputs[j]))

        for downsample_layer in self.downsample_layers:
            outputs.append(downsample_layer(outputs[-1]))

        return outputs


class RetinaNetPredictionHead(nn.Module):
    num_anchors = 9
    def __init__(self):
        super().__init__()
        self.backbone = None
        self.neck = FPN()

        self.boxes_layers = None
        self.class_layers = None

    def _make_layers(self, out_channels):
        layers = []
        for _ in range(4):
            nn.Conv2d(
                in_channels=256, 
                out_channels=256, 
                stride=1)
        nn.Conv2d(
            in_channels=256, 
            out_channels=out_channels)

        nn.ModuleList(nn.Sequential(*layers))
        return 
