import copy
import math
from functools import partial
from typing import Any, Callable, Optional, List

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
# from torchvision._internally_replaced_utils import load_state_dict_from_url


__all__ = [
    'EfficientNet',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
]


model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "expand_ratio={expand_ratio}"
        s += ", kernel={kernel}"
        s += ", stride={stride}"
        s += ", input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))
    
    
def _efficientnet_conf(width_mult: float, depth_mult: float) -> List[MBConvConfig]:
    bneck_conf = partial(
        MBConvConfig, width_mult=width_mult, depth_mult=depth_mult
    )
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    
    return inverted_residual_setting


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels
        self.stride = cnf.stride

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        width_mult: float,
        depth_mult: float,
        stochastic_depth_prob: float = 0.2,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.channels = []

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.inverted_residual_setting = _efficientnet_conf(
            width_mult=width_mult, depth_mult=depth_mult
        )

        # building first layer
        firstconv_output_channels = self.inverted_residual_setting[0].input_channels
        self.firstconv_layer = ConvNormActivation(
            3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in self.inverted_residual_setting)
        stage_block_id = 0
        for cnf in self.inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
                
            # self.channels.append(block_cnf.out_channels)
            self.layers.extend(stage)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.firstconv_layer(inputs)

        outputs = []
        last_x = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if layer.stride == 2:
                outputs.append(last_x)
            elif i == len(self.layers) - 1:
                outputs.append(x)
            last_x = x
        
        del last_x

        return outputs[1:]
    
    def from_pretrained(self, path):
        state_dict = torch.load(path)
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=True)

        try:
            excepted_keys = [
                key for key in list(state_dict)
                if key.startswith('features.8') or key.startswith('classifier')
            ]
            for excepted_key in excepted_keys:
                state_dict.pop(excepted_key)
        except KeyError:
            pass

        self.load_state_dict(state_dict, strict=False)


def efficientnet_b0() -> EfficientNet:
    backbone = EfficientNet(width_mult=1.0, depth_mult=1.0)
    return backbone


def efficientnet_b1() -> EfficientNet:
    backbone = EfficientNet(width_mult=1.0, depth_mult=1.1)
    return backbone


def efficientnet_b2() -> EfficientNet:
    backbone = EfficientNet(width_mult=1.1, depth_mult=1.2)
    return backbone


def efficientnet_b3() -> EfficientNet:
    backbone = EfficientNet(width_mult=1.2, depth_mult=1.4)
    return backbone


def efficientnet_b4() -> EfficientNet:
    backbone = EfficientNet(width_mult=1.4, depth_mult=1.8)
    return backbone


def efficientnet_b5() -> EfficientNet:
    backbone = EfficientNet(
        width_mult=1.6,
        depth_mult=2.2,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    )
    return backbone


def efficientnet_b6() -> EfficientNet:
    backbone = EfficientNet(
        width_mult=1.8,
        depth_mult=2.6,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    )
    return backbone


def efficientnet_b7() -> EfficientNet:
    backbone = EfficientNet(
        width_mult=2.0,
        depth_mult=3.1,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    )
    return backbone
