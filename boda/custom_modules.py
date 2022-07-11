import collections
import math
import re
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class Conv2dDynamicSamePadding(nn.Conv2d):
    """
    Adapted from:
    https://github.com/lukemelas/EfficientNet-PyTorch
    https://github.com/rwightman/pytorch-image-models

    2D Convolutions like TensorFlow, for a dynamic image size.
    The padding is operated in forward function by calculating dynamically.

    Tips for 'SAME' mode padding.
        Given the following:
            i: width or height
            s: stride
            k: kernel size
            d: dilation
            p: padding
        Output after Conv2d:
            o = floor((i+p-((k-1)*d+1))/s+1)
    If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    => p = (i-1)*s+((k-1)*d+1)-i
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(
            iw / sw
        )  # change the output size according to stride

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv2dStaticSamePadding(nn.Conv2d):
    """
    2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
    The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        image_size=None,
        **kwargs
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None

        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """
    2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
    The padding is operated in forward function by calculating dynamically.
    """

    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ) -> None:
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = (
            [self.kernel_size] * 2
            if isinstance(self.kernel_size, int)
            else self.kernel_size
        )
        self.dilation = (
            [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation
        )

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """
    2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
    The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs) -> None:
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = (
            [self.kernel_size] * 2
            if isinstance(self.kernel_size, int)
            else self.kernel_size
        )
        self.dilation = (
            [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation
        )
        assert image_size is not None

        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        return x
