import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchsummary import summary


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes, 
            kernel_size=1, 
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, 
            planes, 
            kernel_size=3,
            stride=stride, 
            padding=1, 
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, 
            planes * self.expansion, 
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        residual = inputs

        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = F.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))

        if self.downsample is not None:
            residual = self.downsample(inputs)
        
        outputs += residual
        outputs = F.relu(outputs)

        return outputs


class ResNet(nn.Module):
    def __init__(self, layers, block=Bottleneck):
        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []

        self.inplanes = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, 
                    planes * block.expansion, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        # Add identity block
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return self

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return outs

    def init_weights(self, path):
        state_dict = torch.load(path)

        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)


def resnet50():
    backbone = ResNet([3, 4, 6, 3])
    # Add downsampling layers until we reach the number we need
    selected_layers = [1, 2, 3]
    # num_layers = max(cfg.selected_layers) + 1
    num_layers = max(selected_layers) + 1
    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone


def resnet101():
    backbone = ResNet([3, 4, 23, 3])
    # Add downsampling layers until we reach the number we need
    selected_layers = [1, 2, 3]
    # num_layers = max(cfg.selected_layers) + 1
    num_layers = max(selected_layers) + 1
    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone

