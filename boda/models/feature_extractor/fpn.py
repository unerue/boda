from typing import List, Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FeaturePyramidNetworks(nn.Module):
    """Pyramid Feature Networks

    Example::
        >>> backbone = resnet101()
        >>> neck = FeaturePyramidNetworks(backbone.channels, [1, 2, 3])
        >>> print(neck.channels, neck.selected_layers)
    """
    def __init__(
        self,
        in_channels: Sequence[int] = [256, 512, 1024, 2048],
        selected_layers: Sequence[int] = [1, 2, 3],
        out_channels: int = 256,
        extra_layers: bool = False,
        num_extra_predict_layers: int = 2,
        **kwargs
    ) -> None:
        """
        Args:
            channels (:obj:`List[int]`): out channels from backbone
            selected_layers (:obj:`List[int]`): to use selected backbone layers
            out_channels (:obj:`int`):
            num_extra_predict_layers (:obj:`int`): make extra predict layers for training
            num_downsamples: (:obj:`int`): use predict layers does not training
        """
        super().__init__()
        self.in_channels = [in_channels[i] for i in selected_layers]
        self.selected_layers = selected_layers
        self.selected_backbones = selected_layers

        self.extra_layers = extra_layers
        self.num_extra_layers = 0
        self.num_extra_predict_layers = num_extra_predict_layers

        self.selected_layers = \
            list(range(len(self.selected_layers) + self.num_extra_predict_layers))

        self.lateral_layers = nn.ModuleList()
        for _in_channels in reversed(self.in_channels):
            self.lateral_layers.append(
                nn.Conv2d(
                    _in_channels,
                    out_channels,
                    kernel_size=kwargs.get('lateral_kernel_size', 1),
                    stride=kwargs.get('lateral_stride', 1),
                    padding=kwargs.get('lateral_padding', 0)
                )
            )

        self.predict_layers = nn.ModuleList()
        for _ in self.in_channels:
            self.predict_layers.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=kwargs.get('', 3),
                    stride=kwargs.get('', 1),
                    padding=kwargs.get('', 1),
                )
            )

        if self.num_extra_predict_layers > 0:
            self.extra_layers = nn.ModuleList([
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ) for _ in range(self.num_extra_predict_layers)
            ])
            # self.channels.append(self.out_channels)

        self.channels = [out_channels] * len(self.selected_layers)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        Args:
            inputs (:obj:`FloatTensor[B, C, H, W]`)

        Returns:
            outputs (:obj:`List[FloatTensor[B, C, H, W]]`)
        """
        device = inputs[0].device
        inputs = [inputs[i] for i in self.selected_backbones]

        x = torch.zeros(1, device=device)
        outputs = [x for _ in range(len(inputs))]

        i = len(inputs)
        for lateral_layer in self.lateral_layers:
            i -= 1
            if i < len(inputs) - 1:
                _, _, h, w = inputs[i].size()
                x = F.interpolate(
                    x, size=(h, w), mode='bilinear', align_corners=False)

            x = x + lateral_layer(inputs[i])
            outputs[i] = x

        i = len(inputs)
        for predict_layer in self.predict_layers:
            i -= 1
            outputs[i] = F.relu(predict_layer(outputs[i]))

        if self.extra_layers:
            for extra_layer in self.extra_layers:
                outputs.append(extra_layer(outputs[-1]))

        elif self.num_extra_predict_layers > 0:
            for _ in range(self.num_extra_predict_layers):
                outputs.append(self.predict_layers[-1](outputs[-1]))

        return outputs
