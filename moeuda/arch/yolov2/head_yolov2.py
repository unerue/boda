from typing import Tuple, List, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


class Yolov2PredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = None
        self.num_classes = 20
        self.num_anchors = 5


        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, 
                out_channels=1024, 
                kernel_size=3, 
            ), 
            nn.Conv2d(
                in_channels=1024, 
                out_channels=1024, 
                kernel_size=3, 
            ))
            
            # conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True),
            # conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True))

        self.downsample = nn.Conv2d(
            in_channels=512, 
            out_channels=64, 
            kernel_size=1)
        
            # conv_bn_leaky(512, 64, kernel_size=1, return_module=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1280,
                out_channels=1024,
                kernel_size=3,
            ), 
            nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1))
            
            # conv_bn_leaky(1280, 1024, kernel_size=3, return_module=True),
            
            

        self.reorg_layer = ReorgLayer()
    
    def forward(self, x):
        outputs = self.backbone(x)
        shortcut = self.reorg_layer(self.downsample(outputs[-2]))

        output = self.conv1(outputs[-1])
        output = torch.cat([shortcut, output], dim=1)
        output = self.conv2(output)

        return output