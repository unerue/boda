from typing import List
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _check_inputs(inputs: List[Tensor]):
    for image in inputs:
        if isinstance(image, Tensor):
            if image.dim() != 3:
                raise ValueError(f'images is expected to be 3d tensors of shape [C, H, W] {image.shape}')
    
        else:
            raise ValueError('Expected image to be Tensor.')


class Conv2d1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)