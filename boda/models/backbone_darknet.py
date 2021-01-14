import math
from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import Backbone


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    
    Source from:
    https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DarkNet19(Backbone):
    """DarkNet19 for YOLOv1, v2 backbone
    """
    def __init__(
        self,
        backbone_structure: List,
        in_channels: int = 3,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        self.use_bn = use_bn

        self._in_channels = in_channels
        self.channels = []
        self.layers = nn.ModuleList()

        for structure in backbone_structure:
            self._make_layers(structure)

        # self.init_weights()

    def forward(self, inputs: Tensor) -> List[Tensor]:
        outputs = []
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)

        return outputs

    def _make_layers(self, layers: List):
        _layers = []
        for v in layers:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]

            if v == 'M':
                _layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if kwargs is None:
                    # kwargs = {'kernel_size': 3, 'padding': 1}
                    kwargs = {'kernel_size': 3}

                _layers += [
                    Conv2dDynamicSamePadding(
                        in_channels=self._in_channels,
                        out_channels=v,
                        bias=False,
                        **kwargs)]

                if self.use_bn:
                    _layers += [
                        nn.BatchNorm2d(v),
                        nn.LeakyReLU(0.1)]
                else:
                    _layers += [nn.LeakyReLU(0.1)]

                self._in_channels = v

        self.channels.append(self._in_channels)
        self.layers.append(nn.Sequential(*_layers))

    def init_weights(self):
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

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        raise NotImplementedError


DARKNET_STRUCTURES = {
    'darknet-base': [
        [(64, {'kernel_size': 7, 'stride': 2, 'padding': 3}), 'M'],
        [192, 'M'],
        [(128, {'kernel_size': 1}), 256, (256, {'kernel_size': 1}), 512, 'M'],
        [
            (256, {'kernel_size': 1}), 512, 
            (256, {'kernel_size': 1}), 512, 
            (256, {'kernel_size': 1}), 512, 
            (256, {'kernel_size': 1}), 512,
            (512, {'kernel_size': 1}), 1024, 'M'],
        [(512, {'kernel_size': 1}), 1024, (512, {'kernel_size': 1}), 'M', 1024, 1024, 1024]],

    'darknet-tiny': [],
}


DARKNET_STRUCTURES = {
    'darknet-base': [
        [(64, {'kernel_size': 7, 'stride': 2}), 'M'],
        [192, 'M'],
        [(128, {'kernel_size': 1}), 256, (256, {'kernel_size': 1}), 512, 'M'],
        [(256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512,
         (256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512,
         (512, {'kernel_size': 1}), 1024, 'M'],
        [(512, {'kernel_size': 1}), 1024, (512, {'kernel_size': 1}), 'M', 1024, 1024, 1024]],

    'darknet-tiny': [],
}


def darknet(structure_or_name: str = 'darknet-base', pretrained: bool = False, **kwargs):
    if isinstance(structure_or_name, str):
        backbone = DarkNet19(DARKNET_STRUCTURES[structure_or_name], **kwargs)

    return backbone
