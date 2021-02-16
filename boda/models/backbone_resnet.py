from typing import Tuple, List, Optional, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# TODO: BACKBONE_ARCHIVE_MAP or _MAPS? or ARCHIVES?
BACKBONE_ARCHIVE_MAP = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class Conv2d1x1(nn.Sequential):
    """1x1 convolution"""
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        )


class Conv2d3x3(nn.Sequential):
    """3x3 convolution with padding"""
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=3, stride=stride,
                padding=dilation, groups=groups, bias=False, dilation=dilation)
        )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = Conv2d3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = Conv2d1x1(in_planes, planes)
        self.bn1 = norm_layer(planes)

        self.conv2 = Conv2d3x3(
            planes,
            planes,
            stride=stride)
        self.bn2 = norm_layer(planes)

        self.conv3 = Conv2d1x1(
            planes,
            planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs) -> Tensor:
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

        # self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        self.backbone_modules = [m for m in self.modules()]

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
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        # Add identity block
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.bn(inputs)
        inputs = self.relu(inputs)
        inputs = self.maxpool(inputs)

        outputs = []
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)

        return outputs

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)

    def from_pretrained(self, path):
        state_dict = torch.load(path)

        try:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
        except KeyError:
            pass

        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(state_dict, strict=False)


def resnet18():
    backbone = ResNet([2, 2, 2, 2], BasicBlock)
    return backbone


def resnet34():
    backbone = ResNet([3, 4, 6, 3], BasicBlock)
    return backbone


def resnet50(pretrained: bool = False):
    backbone = ResNet([3, 4, 6, 3])
    # Add downsampling layers until we reach the number we need
    # selected_layers = [1, 2, 3]
    # num_layers = max(cfg.selected_layers) + 1
    # num_layers = max(selected_layers) + 1
    # while len(backbone.layers) < num_layers:
    #     backbone.add_layer()

    return backbone


def resnet101(pretrained: bool = False):
    backbone = ResNet([3, 4, 23, 3])
    # backbone.from_pretrained('cache/backbones/resnet101_reducedfc.pth')
    # Add downsampling layers until we reach the number we need
    # selected_layers = [1, 2, 3]
    # # num_layers = max(cfg.selected_layers) + 1
    # num_layers = max(selected_layers) + 1
    # while len(backbone.layers) < num_layers:
    #     backbone.add_layer()

    return backbone

