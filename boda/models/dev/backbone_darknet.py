import math
from typing import Tuple, List, Dict, Optional, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import Backbone
from ..modules import Conv2dDynamicSamePadding


class DarkNet(Backbone):
    """DarkNet19 for YOLOv1, v2 backbone
    """
    def __init__(
        self,
        backbone_structure: List,
        in_channels: int = 3,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super().__init__()
        self.norm_layer = norm_layer

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

                if self.norm_layer is not None:
                    _layers += [
                        self.norm_layer(v),
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


# DARKNET_STRUCTURES = {
#     'darknet-base': [
#         [(64, {'kernel_size': 7, 'stride': 2, 'padding': 3}), 'M'],
#         [192, 'M'],
#         [(128, {'kernel_size': 1}), 256, (256, {'kernel_size': 1}), 512, 'M'],
#         [
#             (256, {'kernel_size': 1}), 512, 
#             (256, {'kernel_size': 1}), 512, 
#             (256, {'kernel_size': 1}), 512, 
#             (256, {'kernel_size': 1}), 512,
#             (512, {'kernel_size': 1}), 1024, 'M'],
#         [(512, {'kernel_size': 1}), 1024, (512, {'kernel_size': 1}), 'M', 1024, 1024, 1024]],

#     'darknet-tiny': [],
# }


# DARKNET_STRUCTURES = {
#     'darknet-base': [
#         [(64, {'kernel_size': 7, 'stride': 2}), 'M'],
#         [192, 'M'],
#         [(128, {'kernel_size': 1}), 256, (256, {'kernel_size': 1}), 512, 'M'],
#         [(256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512,
#          (256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512,
#          (512, {'kernel_size': 1}), 1024, 'M'],
#         [(512, {'kernel_size': 1}), 1024, (512, {'kernel_size': 1}), 'M', 1024, 1024, 1024]],

#     'darknet-tiny': [],
# }

DARKNET_STRUCTURES = {
    'darknet-base': [
        [(64, {'kernel_size': 7, 'stride': 2}), 'M'],
        [192, 'M'],
        [(128, {'kernel_size': 1}), 256, (256, {'kernel_size': 1}), 512, 'M'],
        [(256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512,
         (256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512,
         (512, {'kernel_size': 1}), 1024, 'M'],
        [(512, {'kernel_size': 1}), 1024, (512, {'kernel_size': 1}), 'M', 1024]],
}


def darknet(structure_or_name: str = 'darknet-base', pretrained: bool = False, **kwargs):
    if isinstance(structure_or_name, str):
        backbone = DarkNet(DARKNET_STRUCTURES[structure_or_name], norm_layer=nn.BatchNorm2d, **kwargs)

    return backbone
