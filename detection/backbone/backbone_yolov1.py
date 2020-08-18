import math
from typing import List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TinyBlock(nn.Module):
    expansion = 2
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling=True):
        super().__init__()
        self.pooling = pooling
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1, 
            padding=3//2,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky(x)
        if self.pooling:
            x = self.mp(x)

        return x


class Darknet(nn.Module):
    def __init__(self, num_layers=9, block=TinyBlock):
        super().__init__()
        self.in_channels = 3
        self.layers = nn.ModuleList()
        self._make_layers(block, num_layers)
        self.fc = nn.Linear(256*3*3, 1470)

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        
        return outs

    def _make_layers(self, block, num_layers):
        out_channels = 16
        layers = []
        layers.append(block(self.in_channels, out_channels))
        
        pooling = True
        for i in range(3, num_layers):
            self.in_channels = out_channels
            out_channels = out_channels * block.expansion
            layers.append(block(self.in_channels, out_channels, pooling=pooling))
            if (i+1) == num_layers:
                pooling = False
            
        layers.append(block(out_channels, 256, pooling=pooling))
        self.layers.append(nn.Sequential(*layers))
    
    def init_weights(self):
        for m in self.layers:
            if type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal(m.weight)


def darknet9(config: Dict = None):
    model = Darknet()

    model.init_weights()
    
    if isinstance(config, dict):
        if config.backbone.pretrained:
            model.load_state_dict(torch.load(config.backbone.path))

    return model


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class DarkNet(nn.Module):
    def __init__(self, num_classes=1000, conv_only=False, bn=True, init_weight=True):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        # Make layers
        # self.features = self._make_layers()
        self._make_layers()
        if not conv_only:
            self.fc = self._make_fc_layers()

        # Initialize weights
        if init_weight:
            self._initialize_weights()

        self.conv_only = conv_only

    def forward(self, x):
        # x = self.features(x)
        for layer in self.layers:
            x = layer(x)

        if not self.conv_only:
            x = self.fc(x)

        return x

    def _make_layers(self):
        self.layers.append(nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)))

        self.layers.append(nn.Sequential(
            nn.Conv2d(192, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)))

        self.layers.append(nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)))

        self.layers.append(nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)))

    def _make_fc_layers(self):
        fc = nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, 1000)
        )
        return fc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def darknet21():
    return DarkNet().features