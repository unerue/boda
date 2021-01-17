import os
import math
import itertools
import functools
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Dict, Any, Callable, TypeVar, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import Neck, Head, Model
from .configuration_yolact import YolactConfig
from .backbone_resnet import resnet101


class YolactPredictNeck(Neck):
    """Prediction Neck for YOLACT

    Args:
        config (:class:`YolactConfig`):
        channels (:obj:`List[int]`): list of in_channels from backbone
    """
    def __init__(self, config, channels: List[int]) -> None:
        super().__init__()
        self.config = config
        self.selected_layers = list(range(len(config.selected_layers)
                                + config.num_downsamples))
        self.channels = [config.fpn_out_channels] * len(self.selected_layers)

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
        scales (:obj:):
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
        _, prior_boxes = self.prior_box.generate(h, w, inputs.device)

        return_dict = {
            'boxes': boxes,
            'masks': masks,
            'scores': scores,
            'prior_boxes': prior_boxes,
        }

        return return_dict


class YolactPretrained(Model):
    config_class = YolactConfig
    base_model_prefix = 'yolact'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = YolactModel(config)
        # model.state_dict(torch.load('test.pth'))
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


class YolactModel(YolactPretrained):
    """
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝

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
        """
        """
        super().__init__(config)
        self.config = config
        self.use_semantic_segmentation = config.use_semantic_segmentation

        # TODO: rename backbone, neck, head layers
        if backbone is None:
            self.backbone = resnet101()
            num_layers = max(config.selected_layers) + 1
            while len(self.backbone.layers) < num_layers:
                self.backbone.add_layer()

        self.freeze_bn(True)

        self.config.mask_dim = config.mask_size**2

        in_channels = self.backbone.channels[config.proto_src]
        in_channels += config.num_grids

        if neck is None:
            self.neck = YolactPredictNeck(
                config, [self.backbone.channels[i] for i in config.selected_layers])

        in_channels = config.fpn_out_channels
        in_channels += config.num_grids

        self.proto_layer = ProtoNet(
            config, in_channels, config.proto_net, include_last_relu=False)

        self.config.mask_dim = self.proto_layer.channels[-1]

        self.head_layers = nn.ModuleList()
        self.config.num_heads = len(config.selected_layers)
        for i, j in enumerate(self.neck.selected_layers):
            parent = None
            if i > 0:
                parent = self.head_layers[0]

            head_layer = YolactPredictHead(
                config,
                self.neck.channels[j],
                self.neck.channels[j],
                aspect_ratios=config.aspect_ratios[i],
                scales=config.scales[i],
                parent=parent,
                index=i)

            self.head_layers.append(head_layer)

        self.semantic_layer = nn.Conv2d(self.neck.channels[0], config.num_classes-1, kernel_size=1)
        self.init_weights('cache/resnet50-19c8e357.pth')

    def init_weights(self, path):
        self.backbone.from_pretrained(path)

        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, inputs: List[Tensor]) -> Dict[str, List[Tensor]]:
        """
        Args:
            inputs (:obj:`List[FloatTensor[B, C, H, W]]`): 

        `check_inputs` returns :obj:`FloatTensor[B, C, H, W]`.
        `backbone` returns :obj:`List[FloatTensor[B, C, H, W]`.

        Returns:
            return_dict (:obj:`Dict[str, Tensor]`):
                `boxes` (:obj:`FloatTensor[B, N*S, 4]`): B is the number of batch size, S is the number of selected_layers
                `masks` (:obj:`FloatTensor[B, N*S, P]`): P is the number of prototypes
                `scores` (:obj:`FloatTensor[B, N*S, C]`): C is the number of classes with background e.g. 80 + 1
                `priors` (:obj:`FloatTensor[N, 4]`):
                `prototype_masks` (:obj:`FloatTensor[B, H, W, P]`):
                `semantic_masks` (:obj:`FloatTensor[B, C, H, W]`):
        """
        inputs = self.check_inputs(inputs)
        self.config.device = inputs.device

        self.config.size = (inputs.size(2), inputs.size(3))

        outputs = self.backbone(inputs)
        outputs = [outputs[i] for i in self.config.selected_layers]
        outputs = self.neck(outputs)

        return_dict = defaultdict(list)
        # TODO: self.neck.selected_layer...
        for i, layer in zip(self.neck.selected_layers, self.head_layers):
            if layer is not self.head_layers[0]:
                layer.parent = [self.head_layers[0]]

            output = layer(outputs[i])

            for k, v in output.items():
                return_dict[k].append(v)

        for k, v in return_dict.items():
            return_dict[k] = torch.cat(v, dim=-2)

        proto_masks = self.proto_layer(outputs[0])
        proto_masks = F.relu(proto_masks)
        proto_masks = proto_masks.permute(0, 2, 3, 1).contiguous()
        return_dict['proto_masks'] = proto_masks

        if self.training:
            return_dict['semantic_masks'] = self.semantic_layer(outputs[0])
            return return_dict
        else:
            return_dict['scores'] = F.softmax(return_dict['scores'], dim=-1)
            return return_dict
