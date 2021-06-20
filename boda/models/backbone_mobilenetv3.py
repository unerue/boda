import torch

from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

from torchvision.models.utils import load_state_dict_from_url
from .backbone_mobilenetv2 import _make_divisible, ConvBNActivation
# from torchsummary import summary


__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:
    def __init__(
        self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, 
        use_se: bool, activation: str, stride: int, dilation: int, width_mult: float
    ) -> None:
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    def __init__(
        self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation
    ) -> None:
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1
        self.channels = []

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V3 main class
        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []
        # Small
        self.channels = []
        # self.channels = [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96, 576]
        # self.channels = [16, 16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 960]

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(
            3, firstconv_output_channels, kernel_size=3, stride=2,
            norm_layer=norm_layer, activation_layer=nn.Hardswish)
        )
        self.channels.append(firstconv_output_channels)

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
            self.channels.append(cnf.input_channels)

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(
            lastconv_input_channels, lastconv_output_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=nn.Hardswish)
        )
        self.channels.append(lastconv_output_channels)

        # self.features = nn.Sequential(*layers)
        self.features = nn.ModuleList()
        for layer in layers:
            self.features.append(layer)

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(lastconv_output_channels, last_channel),
        #     nn.Hardswish(inplace=True),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.Linear(last_channel, num_classes),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.zeros_(m.bias)

        # self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _forward_impl(self, x: Tensor) -> Tensor:
        # x = self.features(x)
        outputs = []
        # for feat in self.features:
        for (i, feat) in enumerate(self.features):
            x = feat(x)
            outputs.append(x)
            print(i, x.size())

        # print(self.channels)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        # for k, v in self.state_dict().items():
        #     print(k, v.size())
        # print()
        state_dict = torch.load(path)
        # for k, v in state_dict.items():
        #     print(k, v.size())
        # print()
        state_dict.pop('classifier.0.weight')
        state_dict.pop('classifier.0.bias')
        state_dict.pop('classifier.3.weight')
        state_dict.pop('classifier.3.bias')

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx-1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # for k, v in state_dict.items():
        #     print(k, v.size())
        self.load_state_dict(state_dict, strict=False)
        print('Loaded pretrained weights for backbone...')


def _mobilenet_v3_conf(arch: str, params: Dict[str, Any]):
    # non-public config parameters
    reduce_divider = 2 if params.pop('_reduced_tail', False) else 1
    dilation = 2 if params.pop('_dilated', False) else 1
    width_mult = params.pop('_width_mult', 1.0)

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _mobilenet_v3_model(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


# if __name__ == '__main__':
#     model = mobilenet_v3_small(pretrained=False).to('cuda')
#     print(summary(model, input_data=(3, 550, 550), verbose=0))
    # for k, v in model.state_dict().items():
    #     print(k, v.size())

    # for i, layer in enumerate(model.features):
    #     print(i, layer.named_modules())

"""
0 torch.Size([2, 16, 275, 275]) [6, 7, 10]
1 torch.Size([2, 16, 275, 275]) [6, 10, 12]
2 torch.Size([2, 24, 138, 138])
3 torch.Size([2, 24, 138, 138])
4 torch.Size([2, 40, 69, 69])
5 torch.Size([2, 40, 69, 69])
6 torch.Size([2, 40, 69, 69]) * ?
7 torch.Size([2, 80, 35, 35]) *
8 torch.Size([2, 80, 35, 35])
9 torch.Size([2, 80, 35, 35])
10 torch.Size([2, 80, 35, 35]) * ?
11 torch.Size([2, 112, 35, 35])
12 torch.Size([2, 112, 35, 35]) ?
13 torch.Size([2, 160, 18, 18])
14 torch.Size([2, 160, 18, 18])
15 torch.Size([2, 160, 18, 18])
16 torch.Size([2, 960, 18, 18])

0 torch.Size([2, 16, 275, 275]) [3, 4, 6]
1 torch.Size([2, 16, 138, 138]) [3, 6, 8]
2 torch.Size([2, 24, 69, 69])
3 torch.Size([2, 24, 69, 69]) * ?
4 torch.Size([2, 40, 35, 35]) *
5 torch.Size([2, 40, 35, 35])
6 torch.Size([2, 40, 35, 35]) * ?
7 torch.Size([2, 48, 35, 35])
8 torch.Size([2, 48, 35, 35]) ?
9 torch.Size([2, 96, 18, 18])
10 torch.Size([2, 96, 18, 18])
11 torch.Size([2, 96, 18, 18])
12 torch.Size([2, 576, 18, 18])

"""
# features.0.0.weight torch.Size([16, 3, 3, 3])
# features.0.1.weight torch.Size([16])
# features.0.1.bias torch.Size([16])
# features.0.1.running_mean torch.Size([16])
# features.0.1.running_var torch.Size([16])
# features.0.1.num_batches_tracked torch.Size([])
# features.1.block.0.0.weight torch.Size([16, 1, 3, 3])
# features.1.block.0.1.weight torch.Size([16])
# features.1.block.0.1.bias torch.Size([16])
# features.1.block.0.1.running_mean torch.Size([16])
# features.1.block.0.1.running_var torch.Size([16])
# features.1.block.0.1.num_batches_tracked torch.Size([])
# features.1.block.1.fc1.weight torch.Size([8, 16, 1, 1])
# features.1.block.1.fc1.bias torch.Size([8])
# features.1.block.1.fc2.weight torch.Size([16, 8, 1, 1])
# features.1.block.1.fc2.bias torch.Size([16])
# features.1.block.2.0.weight torch.Size([16, 16, 1, 1])
# features.1.block.2.1.weight torch.Size([16])
# features.1.block.2.1.bias torch.Size([16])
# features.1.block.2.1.running_mean torch.Size([16])
# features.1.block.2.1.running_var torch.Size([16])
# features.1.block.2.1.num_batches_tracked torch.Size([])
# features.2.block.0.0.weight torch.Size([72, 16, 1, 1])
# features.2.block.0.1.weight torch.Size([72])
# features.2.block.0.1.bias torch.Size([72])
# features.2.block.0.1.running_mean torch.Size([72])
# features.2.block.0.1.running_var torch.Size([72])
# features.2.block.0.1.num_batches_tracked torch.Size([])
# features.2.block.1.0.weight torch.Size([72, 1, 3, 3])
# features.2.block.1.1.weight torch.Size([72])
# features.2.block.1.1.bias torch.Size([72])
# features.2.block.1.1.running_mean torch.Size([72])
# features.2.block.1.1.running_var torch.Size([72])
# features.2.block.1.1.num_batches_tracked torch.Size([])
# features.2.block.2.0.weight torch.Size([24, 72, 1, 1])
# features.2.block.2.1.weight torch.Size([24])
# features.2.block.2.1.bias torch.Size([24])
# features.2.block.2.1.running_mean torch.Size([24])
# features.2.block.2.1.running_var torch.Size([24])
# features.2.block.2.1.num_batches_tracked torch.Size([])
# features.3.block.0.0.weight torch.Size([88, 24, 1, 1])
# features.3.block.0.1.weight torch.Size([88])
# features.3.block.0.1.bias torch.Size([88])
# features.3.block.0.1.running_mean torch.Size([88])
# features.3.block.0.1.running_var torch.Size([88])
# features.3.block.0.1.num_batches_tracked torch.Size([])
# features.3.block.1.0.weight torch.Size([88, 1, 3, 3])
# features.3.block.1.1.weight torch.Size([88])
# features.3.block.1.1.bias torch.Size([88])
# features.3.block.1.1.running_mean torch.Size([88])
# features.3.block.1.1.running_var torch.Size([88])
# features.3.block.1.1.num_batches_tracked torch.Size([])
# features.3.block.2.0.weight torch.Size([24, 88, 1, 1])
# features.3.block.2.1.weight torch.Size([24])
# features.3.block.2.1.bias torch.Size([24])
# features.3.block.2.1.running_mean torch.Size([24])
# features.3.block.2.1.running_var torch.Size([24])
# features.3.block.2.1.num_batches_tracked torch.Size([])
# features.4.block.0.0.weight torch.Size([96, 24, 1, 1])
# features.4.block.0.1.weight torch.Size([96])
# features.4.block.0.1.bias torch.Size([96])
# features.4.block.0.1.running_mean torch.Size([96])
# features.4.block.0.1.running_var torch.Size([96])
# features.4.block.0.1.num_batches_tracked torch.Size([])
# features.4.block.1.0.weight torch.Size([96, 1, 5, 5])
# features.4.block.1.1.weight torch.Size([96])
# features.4.block.1.1.bias torch.Size([96])
# features.4.block.1.1.running_mean torch.Size([96])
# features.4.block.1.1.running_var torch.Size([96])
# features.4.block.1.1.num_batches_tracked torch.Size([])
# features.4.block.2.fc1.weight torch.Size([24, 96, 1, 1])
# features.4.block.2.fc1.bias torch.Size([24])
# features.4.block.2.fc2.weight torch.Size([96, 24, 1, 1])
# features.4.block.2.fc2.bias torch.Size([96])
# features.4.block.3.0.weight torch.Size([40, 96, 1, 1])
# features.4.block.3.1.weight torch.Size([40])
# features.4.block.3.1.bias torch.Size([40])
# features.4.block.3.1.running_mean torch.Size([40])
# features.4.block.3.1.running_var torch.Size([40])
# features.4.block.3.1.num_batches_tracked torch.Size([])
# features.5.block.0.0.weight torch.Size([240, 40, 1, 1])
# features.5.block.0.1.weight torch.Size([240])
# features.5.block.0.1.bias torch.Size([240])
# features.5.block.0.1.running_mean torch.Size([240])
# features.5.block.0.1.running_var torch.Size([240])
# features.5.block.0.1.num_batches_tracked torch.Size([])
# features.5.block.1.0.weight torch.Size([240, 1, 5, 5])
# features.5.block.1.1.weight torch.Size([240])
# features.5.block.1.1.bias torch.Size([240])
# features.5.block.1.1.running_mean torch.Size([240])
# features.5.block.1.1.running_var torch.Size([240])
# features.5.block.1.1.num_batches_tracked torch.Size([])
# features.5.block.2.fc1.weight torch.Size([64, 240, 1, 1])
# features.5.block.2.fc1.bias torch.Size([64])
# features.5.block.2.fc2.weight torch.Size([240, 64, 1, 1])
# features.5.block.2.fc2.bias torch.Size([240])
# features.5.block.3.0.weight torch.Size([40, 240, 1, 1])
# features.5.block.3.1.weight torch.Size([40])
# features.5.block.3.1.bias torch.Size([40])
# features.5.block.3.1.running_mean torch.Size([40])
# features.5.block.3.1.running_var torch.Size([40])
# features.5.block.3.1.num_batches_tracked torch.Size([])
# features.6.block.0.0.weight torch.Size([240, 40, 1, 1])
# features.6.block.0.1.weight torch.Size([240])
# features.6.block.0.1.bias torch.Size([240])
# features.6.block.0.1.running_mean torch.Size([240])
# features.6.block.0.1.running_var torch.Size([240])
# features.6.block.0.1.num_batches_tracked torch.Size([])
# features.6.block.1.0.weight torch.Size([240, 1, 5, 5])
# features.6.block.1.1.weight torch.Size([240])
# features.6.block.1.1.bias torch.Size([240])
# features.6.block.1.1.running_mean torch.Size([240])
# features.6.block.1.1.running_var torch.Size([240])
# features.6.block.1.1.num_batches_tracked torch.Size([])
# features.6.block.2.fc1.weight torch.Size([64, 240, 1, 1])
# features.6.block.2.fc1.bias torch.Size([64])
# features.6.block.2.fc2.weight torch.Size([240, 64, 1, 1])
# features.6.block.2.fc2.bias torch.Size([240])
# features.6.block.3.0.weight torch.Size([40, 240, 1, 1])
# features.6.block.3.1.weight torch.Size([40])
# features.6.block.3.1.bias torch.Size([40])
# features.6.block.3.1.running_mean torch.Size([40])
# features.6.block.3.1.running_var torch.Size([40])
# features.6.block.3.1.num_batches_tracked torch.Size([])
# features.7.block.0.0.weight torch.Size([120, 40, 1, 1])
# features.7.block.0.1.weight torch.Size([120])
# features.7.block.0.1.bias torch.Size([120])
# features.7.block.0.1.running_mean torch.Size([120])
# features.7.block.0.1.running_var torch.Size([120])
# features.7.block.0.1.num_batches_tracked torch.Size([])
# features.7.block.1.0.weight torch.Size([120, 1, 5, 5])
# features.7.block.1.1.weight torch.Size([120])
# features.7.block.1.1.bias torch.Size([120])
# features.7.block.1.1.running_mean torch.Size([120])
# features.7.block.1.1.running_var torch.Size([120])
# features.7.block.1.1.num_batches_tracked torch.Size([])
# features.7.block.2.fc1.weight torch.Size([32, 120, 1, 1])
# features.7.block.2.fc1.bias torch.Size([32])
# features.7.block.2.fc2.weight torch.Size([120, 32, 1, 1])
# features.7.block.2.fc2.bias torch.Size([120])
# features.7.block.3.0.weight torch.Size([48, 120, 1, 1])
# features.7.block.3.1.weight torch.Size([48])
# features.7.block.3.1.bias torch.Size([48])
# features.7.block.3.1.running_mean torch.Size([48])
# features.7.block.3.1.running_var torch.Size([48])
# features.7.block.3.1.num_batches_tracked torch.Size([])
# features.8.block.0.0.weight torch.Size([144, 48, 1, 1])
# features.8.block.0.1.weight torch.Size([144])
# features.8.block.0.1.bias torch.Size([144])
# features.8.block.0.1.running_mean torch.Size([144])
# features.8.block.0.1.running_var torch.Size([144])
# features.8.block.0.1.num_batches_tracked torch.Size([])
# features.8.block.1.0.weight torch.Size([144, 1, 5, 5])
# features.8.block.1.1.weight torch.Size([144])
# features.8.block.1.1.bias torch.Size([144])
# features.8.block.1.1.running_mean torch.Size([144])
# features.8.block.1.1.running_var torch.Size([144])
# features.8.block.1.1.num_batches_tracked torch.Size([])
# features.8.block.2.fc1.weight torch.Size([40, 144, 1, 1])
# features.8.block.2.fc1.bias torch.Size([40])
# features.8.block.2.fc2.weight torch.Size([144, 40, 1, 1])
# features.8.block.2.fc2.bias torch.Size([144])
# features.8.block.3.0.weight torch.Size([48, 144, 1, 1])
# features.8.block.3.1.weight torch.Size([48])
# features.8.block.3.1.bias torch.Size([48])
# features.8.block.3.1.running_mean torch.Size([48])
# features.8.block.3.1.running_var torch.Size([48])
# features.8.block.3.1.num_batches_tracked torch.Size([])
# features.9.block.0.0.weight torch.Size([288, 48, 1, 1])
# features.9.block.0.1.weight torch.Size([288])
# features.9.block.0.1.bias torch.Size([288])
# features.9.block.0.1.running_mean torch.Size([288])
# features.9.block.0.1.running_var torch.Size([288])
# features.9.block.0.1.num_batches_tracked torch.Size([])
# features.9.block.1.0.weight torch.Size([288, 1, 5, 5])
# features.9.block.1.1.weight torch.Size([288])
# features.9.block.1.1.bias torch.Size([288])
# features.9.block.1.1.running_mean torch.Size([288])
# features.9.block.1.1.running_var torch.Size([288])
# features.9.block.1.1.num_batches_tracked torch.Size([])
# features.9.block.2.fc1.weight torch.Size([72, 288, 1, 1])
# features.9.block.2.fc1.bias torch.Size([72])
# features.9.block.2.fc2.weight torch.Size([288, 72, 1, 1])
# features.9.block.2.fc2.bias torch.Size([288])
# features.9.block.3.0.weight torch.Size([96, 288, 1, 1])
# features.9.block.3.1.weight torch.Size([96])
# features.9.block.3.1.bias torch.Size([96])
# features.9.block.3.1.running_mean torch.Size([96])
# features.9.block.3.1.running_var torch.Size([96])
# features.9.block.3.1.num_batches_tracked torch.Size([])
# features.10.block.0.0.weight torch.Size([576, 96, 1, 1])
# features.10.block.0.1.weight torch.Size([576])
# features.10.block.0.1.bias torch.Size([576])
# features.10.block.0.1.running_mean torch.Size([576])
# features.10.block.0.1.running_var torch.Size([576])
# features.10.block.0.1.num_batches_tracked torch.Size([])
# features.10.block.1.0.weight torch.Size([576, 1, 5, 5])
# features.10.block.1.1.weight torch.Size([576])
# features.10.block.1.1.bias torch.Size([576])
# features.10.block.1.1.running_mean torch.Size([576])
# features.10.block.1.1.running_var torch.Size([576])
# features.10.block.1.1.num_batches_tracked torch.Size([])
# features.10.block.2.fc1.weight torch.Size([144, 576, 1, 1])
# features.10.block.2.fc1.bias torch.Size([144])
# features.10.block.2.fc2.weight torch.Size([576, 144, 1, 1])
# features.10.block.2.fc2.bias torch.Size([576])
# features.10.block.3.0.weight torch.Size([96, 576, 1, 1])
# features.10.block.3.1.weight torch.Size([96])
# features.10.block.3.1.bias torch.Size([96])
# features.10.block.3.1.running_mean torch.Size([96])
# features.10.block.3.1.running_var torch.Size([96])
# features.10.block.3.1.num_batches_tracked torch.Size([])
# features.11.block.0.0.weight torch.Size([576, 96, 1, 1])
# features.11.block.0.1.weight torch.Size([576])
# features.11.block.0.1.bias torch.Size([576])
# features.11.block.0.1.running_mean torch.Size([576])
# features.11.block.0.1.running_var torch.Size([576])
# features.11.block.0.1.num_batches_tracked torch.Size([])
# features.11.block.1.0.weight torch.Size([576, 1, 5, 5])
# features.11.block.1.1.weight torch.Size([576])
# features.11.block.1.1.bias torch.Size([576])
# features.11.block.1.1.running_mean torch.Size([576])
# features.11.block.1.1.running_var torch.Size([576])
# features.11.block.1.1.num_batches_tracked torch.Size([])
# features.11.block.2.fc1.weight torch.Size([144, 576, 1, 1])
# features.11.block.2.fc1.bias torch.Size([144])
# features.11.block.2.fc2.weight torch.Size([576, 144, 1, 1])
# features.11.block.2.fc2.bias torch.Size([576])
# features.11.block.3.0.weight torch.Size([96, 576, 1, 1])
# features.11.block.3.1.weight torch.Size([96])
# features.11.block.3.1.bias torch.Size([96])
# features.11.block.3.1.running_mean torch.Size([96])
# features.11.block.3.1.running_var torch.Size([96])
# features.11.block.3.1.num_batches_tracked torch.Size([])
# features.12.0.weight torch.Size([576, 96, 1, 1])
# features.12.1.weight torch.Size([576])
# features.12.1.bias torch.Size([576])
# features.12.1.running_mean torch.Size([576])
# features.12.1.running_var torch.Size([576])
# features.12.1.num_batches_tracked torch.Size([])
# classifier.0.weight torch.Size([1024, 576])
# classifier.0.bias torch.Size([1024])
# classifier.3.weight torch.Size([1000, 1024])
# classifier.3.bias torch.Size([1000])
