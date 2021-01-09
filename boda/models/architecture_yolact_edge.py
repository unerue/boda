import os
import math
import itertools
import functools
from collections import defaultdict
from collections import OrderedDict
from typing import Tuple, List, Dict, Any, Callable, TypeVar, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import Neck, Head, Model
from .configuration_yolact import YolactConfig
from .backbone_resnet import resnet101


class PredictFlow(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(
            nn.Conv2d(
                in_channels, 2, kernel_size=3, stride=1, padding=1, bias=False)
        )


class Conv2dBnReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1
    ) -> nn.Module:
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ConvBnReLU(nn.Sequential):
    def __init__(self):
        super().__init__()


class Conv2dOnly(nn.Sequential):
    def __init__(self):
        super().__init__()


class Conv2dLeakyReLU(nn.Sequential):
    def __init__(self):
        super().__init__()


class Conv2dBn(nn.Sequential):
    def __init__(self):
        super().__init__()


class ShuffleCat(nn.Module):
    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        a = a.permute(0, 2, 3, 1).contiguous().view(-1, c)
        b = b.permute(0, 2, 3, 1).contiguous().view(-1, c)
        x = torch.cat((a, b), dim=0).transpose(1, 0).contiguous()
        x = x.view(c * 2, n, h, w).permute(1, 0, 2, 3)
        return x


class ShuffleCatChunk(nn.Module):
    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        a = torch.chunk(a, chunks=c, dim=1)
        b = torch.chunk(b, chunks=c, dim=1)
        x = [None] * (c * 2)
        x[::2] = a
        x[1::2] = b
        x = torch.cat(x, dim=1)
        return x


class ShuffleCatAlt(nn.Module):
    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        x = torch.zeros(n, c*2, h, w, dtype=a.dtype, device=a.device)
        x[:, ::2] = a
        x[:, 1::2] = b
        return x


class FlowNetUnwrap(nn.Module):
    def forward(self, preds):
        outs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        flow1, scale1, bias1, flow2, scale2, bias2, flow3, scale3, bias3 = preds

        outs.append((flow1, scale1, bias1))
        outs.append((flow2, scale2, bias2))
        outs.append((flow3, scale3, bias3))
        return outs


class YolactEdgePredictNeck(Neck):
    """Prediction Neck for YOLACT

    Args:
        config (:class:`YolactConfig`): 
        channels (:obj:`List[int]`): list of in_channels from backbone
    """
    def __init__(self, config, channels: List[int]) -> None:
        super().__init__()
        self.config = config
        _selected_layers = list(range(len(config.selected_layers)
                                + config.num_downsamples))
        self.channels = [config.fpn_out_channels] * len(_selected_layers)

        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(
                _in_channels,
                config.fpn_out_channels,
                kernel_size=1) for _in_channels in reversed(channels)])

        self.predict_layers = nn.ModuleList([
            nn.Conv2d(
                config.fpn_out_channels,
                config.fpn_out_channels,
                kernel_size=3,
                padding=config.padding) for _ in channels])

        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(
                config.fpn_out_channels,
                config.fpn_out_channels,
                kernel_size=3,
                stride=2,
                padding=1) for _ in range(config.num_downsamples)])

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        Args:
            inputs (:obj:`FloatTensor[B, C, H, W]`)

        Returns:
            outputs (:obj:`List[FloatTensor[B, C, H, W]]`)
        """
        x = torch.zeros(1, device=self.config.device)
        outputs = [x for _ in range(len(inputs))]

        j = len(inputs)
        for lateral_layer in self.lateral_layers:
            j -= 1
            if j < len(inputs) - 1:
                _, _, h, w = inputs[j].size()
                x = F.interpolate(
                    x, size=(h, w), mode='bilinear', align_corners=False)

            x = x + lateral_layer(inputs[j])
            outputs[j] = x

        j = len(inputs)
        for predict_layer in self.predict_layers:
            j -= 1
            outputs[j] = F.relu(predict_layer(outputs[j]))

        for downsample_layer in self.downsample_layers:
            outputs.append(downsample_layer(outputs[-1]))

        return outputs


# T = TypeVar('T', bound=Callable[..., Any])
def prior_cache(func):
    cache = defaultdict()

    @functools.wraps(func)
    def wrapper(*args):
        k, v = func(*args)
        if k not in cache:
            cache[k] = v
        return k, cache[k]
    return wrapper


class PriorBox:
    """
    Args:
        aspect_ratios (:obj:`List[int]`): 
        scales (:obj:),
        max_size ():
        use_preapply_sqrt ():
        use_pixel_scales ():
        use-square_anchors (:obj:`bool`) default `True`
    """
    def __init__(
        self,
        aspect_ratios: List[int],
        scales: List[float],
        max_size: int,
        use_preapply_sqrt: bool = True,
        use_pixel_scales: bool = True,
        use_square_anchors: bool = True
    ) -> None:
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.use_preapply_sqrt = use_preapply_sqrt
        self.use_pixel_scales = use_pixel_scales
        self.use_square_anchors = use_square_anchors

    @prior_cache
    def generate(
        self,
        h: int,
        w: int,
        device: str = 'cuda'
    ) -> Tuple[Tuple[int], Tensor]:
        """
        Args:
            h (:obj:`int`): feature map size from backbone
            w (:obj:`int`): feature map size from backbone
            device (:obj:`str`): default `cuda`

        Returns
            size (:obj:`Tuple[int]`): feature map size
            prior_boxes (:obj:`FloatTensor[N, 4]`):
        """
        size = (h, w)
        prior_boxes = []
        for j, i in itertools.product(range(h), range(w)):
            x = (i + 0.5) / w
            y = (j + 0.5) / h
            for ratios in self.aspect_ratios:
                for scale in self.scales:
                    for ratio in ratios:
                        if not self.use_preapply_sqrt:
                            ratio = math.sqrt(ratio)

                        if self.use_pixel_scales:
                            _h = scale / ratio / self.max_size[0]
                            _w = scale * ratio / self.max_size[1]
                        else:
                            _h = scale / ratio / h
                            _w = scale * ratio / w

                        if self.use_square_anchors:
                            _h = _w

                        prior_boxes += [x, y, _w, _h]
        
        # TODO: thinking processing to(device)
        prior_boxes = \
            torch.as_tensor(prior_boxes, dtype=torch.float32).view(-1, 4).to(device)
        prior_boxes.requires_grad = False

        return size, prior_boxes


class ProtoNet(nn.Sequential):
    """ProtoNet of YOLACT

    Arguments:
        config (:class:`YolactConfig`)
        in_channels (:obj:`int`):
        layers ():
        include_last_relu (Optional[bool]): 
    """
    def __init__(
        self,
        config,
        in_channels: int,
        layers: List,
        include_last_relu: bool = True
    ) -> None:
        self.config = config
        self.channels = []
        # TODO: mask_layers or _layers?
        mask_layers = OrderedDict()
        for i, v in enumerate(layers):
            if isinstance(v[0], int):
                mask_layers[f'{i}'] = nn.Conv2d(
                    in_channels, v[0], kernel_size=v[1], **v[2])
                self.channels.append(v[0])
            elif v[0] is None:
                mask_layers[f'{i}'] = nn.Upsample(
                    scale_factor=-v[1],
                    mode='bilinear',
                    align_corners=False,
                    **v[2])

        if include_last_relu:
            mask_layers[f'relu{len(mask_layers)+1}'] = nn.ReLU()

        super().__init__(mask_layers)


class HeadBranch(nn.Sequential):
    """
    TODO: all replace
    """
    def __init__(
        self,
        num_extra_layers: int,
        in_channels: int,
        out_channels: int
    ) -> None:
        sub_layers = OrderedDict()
        if num_extra_layers > 0:
            for i in range(num_extra_layers):
                sub_layers[f'{i}'] = [
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=1),
                    nn.ReLU()]

        sub_layers[f'{len(sub_layers)+1}'] = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)

        super().__init__(sub_layers)


class YolactPredictHead(Head):
    """Prediction Head for YOLACT

    Args:
        config (:class:`YolactConfig`): 
        in_channles (:obj:`int`):
        out_channels (:obj:`int`):
        aspect_ratios (:obj:`List[int]`):
        scales ():
        parent ():
        index ():
    """
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        aspect_ratios: List[int],
        scales: List[float],
        parent: nn.Module,
        index: int
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.prior_box = PriorBox(
            aspect_ratios,
            scales,
            config.max_size,
            config.use_preapply_sqrt,
            config.use_pixel_scales,
            config.use_square_anchors)

        self.mask_dim = config.mask_dim
        self.num_priors = sum(len(x)*len(scales) for x in aspect_ratios)
        self.parent = [parent]

        if parent is None:
            if config.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upsample_layers = ProtoNet(
                    config, in_channels, config.extra_head_net)
                out_channels = self.upsample_layers.channels[-1]

            self.bbox_layers = self._make_layer(
                config.num_extra_bbox_layers,
                out_channels,
                self.num_priors * 4)

            self.conf_layers = self._make_layer(
                config.num_extra_conf_layers,
                out_channels,
                self.num_priors * self.num_classes)

            self.mask_layers = self._make_layer(
                config.num_extra_mask_layers,
                out_channels,
                self.num_priors * self.mask_dim)

    def _make_layer(
        self,
        num_extra_layers: int,
        in_channels: int,
        out_channels: int
    ) -> nn.Sequential:
        """
        """
        _predict_layers = []
        if num_extra_layers > 0:
            for _ in range(num_extra_layers):
                _predict_layers += [
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=1),
                    nn.ReLU()]

        _predict_layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

        return nn.Sequential(*_predict_layers)

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            inputs (:obj:`FloatTensor[B, C, H, W]`):

        Returns:
            return_dict (:obj:`Dict[str, Tensor]`):
                `boxes` (:obj:`FloatTensor[B, N, 4]`): B is the number of batch size
                `masks` (:obj:`FloatTensor[B, N, P]`): P is the number of prototypes
                `scores` (:obj:`FloatTensor[B, N, C]`): C is the number of classes with background e.g. 80 + 1
                `priors` (:obj:`FloatTensor[N, 4]`):
        """
        branches = self if self.parent[0] is None else self.parent[0]

        h, w = inputs.size(2), inputs.size(3)

        inputs = branches.upsample_layers(inputs)

        boxes = branches.bbox_layers(inputs)
        masks = branches.mask_layers(inputs)
        scores = branches.conf_layers(inputs)

        boxes = boxes.permute(0, 2, 3, 1).contiguous().view(inputs.size(0), -1, 4)
        masks = masks.permute(0, 2, 3, 1).contiguous().view(inputs.size(0), -1, self.mask_dim)
        masks = torch.tanh(masks)
        scores = scores.permute(0, 2, 3, 1).contiguous().view(inputs.size(0), -1, self.num_classes)
        _, priors = self.prior_box.generate(h, w, inputs.device)

        return_dict = {
            'boxes': boxes,
            'masks': masks,
            'scores': scores,
            'priors': priors,
        }

        return return_dict



class YolactEdgePretrained(Model):
    config_class = YolactConfig
    base_model_prefix = 'yolact'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = YolactModel(config)
        # model.state_dict(torch.load('yolact.pth'))
        return model

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class YolactEdgeModel(YolactEdgePretrained):
    """
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗    ███████╗██████╗  ██████╗ ███████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║       █████╗  ██║  ██║██║  ███╗█████╗  
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║       ██╔══╝  ██║  ██║██║   ██║██╔══╝  
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║       ███████╗██████╔╝╚██████╔╝███████╗
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝       ╚══════╝╚═════╝  ╚═════╝ ╚══════╝

    Args:
        config (:class:`YolactConfig`):
        backbone ()
        neck (:class:`Neck`)
        head (:class:`Head`)
        num_classes (:obj:`int`):

    Examples::
        >>> from boda import YolactConfig, YolactModel

        >>> config = YolactConfig()
        >>> model = YolactModel(config)
    """
    model_name = 'yolact'

    def __init__(
        self,
        config,
        backbone = None,
        neck: Neck = None,
        head: Head = None,
        num_classes: int = None,
        **kwargs
    ) -> None:
        ...