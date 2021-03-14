import os
import math
import itertools
import functools
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Dict, Any, Callable, TypeVar, Union, Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ...base_architecture import Backbone, Neck, Head, Model
from .configuration_yolact import YolactConfig
from ..backbone_resnet import resnet101
from ..neck_fpn import FeaturePyramidNetwork


class YolactPredictNeck(FeaturePyramidNetwork):
    def __init__(
        self,
        config: YolactConfig,
        channels: Sequence[int],
        selected_layers: Sequence[int] = [1, 2, 3],
        fpn_channels: int = 256,
        extra_layers: bool = True,
        num_extra_fpn_layers: int = 2,
    ) -> None:
        super().__init__(
            config,
            channels,
            selected_layers,
            fpn_channels,
            extra_layers,
            num_extra_fpn_layers
        )


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
        max_size: Tuple[int] = (550, 550),
        use_preapply_sqrt: bool = True,
        use_pixel_scales: bool = True,
        use_square_anchors: bool = True
    ) -> None:
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.max_size = max_size
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
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: int = 2,
        mode: str = 'bilinear'
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )


class HeadBranch(Head):
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
        index: int,
        num_classes: int,
        use_preapply_sqrt: bool = True,
        use_pixel_scales: bool = False,
        use_square_anchors: bool = True,
        extra_head_layer_structure: List = [(256, 3, {'padding': 1})],
        num_extra_box_layers: int = 0,
        num_extra_mask_layers: int = 0,
        num_extra_score_layers: int = 0
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.extra_head_layer_structure = extra_head_layer_structure
        self.num_extra_box_layers = num_extra_box_layers
        self.num_extra_mask_layers = num_extra_mask_layers
        self.num_extra_score_layers = num_extra_score_layers

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
            self.upsample_layers = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU()
            )
            out_channels = in_channels

            self.box_layers = self._make_layer(
                self.num_extra_box_layers,
                out_channels,
                self.num_priors * 4
            )

            self.mask_layers = self._make_layer(
                self.num_extra_mask_layers,
                out_channels,
                self.num_priors * self.mask_dim
            )

            self.score_layers = self._make_layer(
                self.num_extra_score_layers,
                out_channels,
                self.num_priors * self.num_classes
            )

    def _make_layer(
        self,
        num_extra_layers: int,
        in_channels: int,
        out_channels: int
    ) -> nn.Sequential:
        """
        """
        _layers = []
        if num_extra_layers > 0:
            for _ in range(num_extra_layers):
                _layers += [
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=1),
                    nn.ReLU()
                ]

        _layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        return nn.Sequential(*_layers)

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            inputs (:obj:`FloatTensor[B, C, H, W]`):

        Returns:
            return_dict (:obj:`Dict[str, Tensor]`):
                `boxes` (:obj:`FloatTensor[B, N, 4]`): B is the number of batch size
                `masks` (:obj:`FloatTensor[B, N, P]`): P is the number of prototypes
                `scores` (:obj:`FloatTensor[B, N, C]`): C is the number of classes with background e.g. 80 + 1
                `prior_boxes` (:obj:`FloatTensor[N, 4]`):
        """
        branches = self if self.parent[0] is None else self.parent[0]

        h, w = inputs.size(2), inputs.size(3)
        inputs = branches.upsample_layers(inputs)

        boxes = branches.box_layers(inputs)
        masks = branches.mask_layers(inputs)
        scores = branches.score_layers(inputs)

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


class YolactPredictHead(nn.Sequential):
    """
    TODO: all replace
    """
    def __init__(
        self,
        config,
        selected_layers,
        in_channels,
        aspect_ratios: int,
        scales: int,
        num_classes: int,
    ) -> None:
        head_layers = []
        for i, j in enumerate(selected_layers):
            parent = None
            if i > 0:
                parent = head_layers[0]

            head_layers.append(HeadBranch(
                config,
                in_channels[j],
                in_channels[j],
                aspect_ratios=aspect_ratios[i],
                scales=scales[i],
                parent=parent,
                index=i,
                num_classes=num_classes
            ))
        super().__init__(*head_layers)


class SemanticSegmentation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size)
        )


class YolactPretrained(Model):
    config_class = YolactConfig
    base_model_prefix = 'yolact'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = YolactModel(config)

        pretrained_file = super().get_pretrained_from_file(
            name_or_path, cache_dir=cls.config_class.cache_dir)
        model.load_weights(pretrained_file)

        return model


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
        config: YolactConfig,
        backbone: Backbone = resnet101,
        neck: Neck = YolactPredictNeck,
        head: Head = YolactPredictHead,
        num_classes: int = None,
        selected_layers: Sequence = [1, 2, 3],
        fpn_channels: int = 256,
        num_grids: int = 0,
        mask_size: int = 16,
        aspect_ratios: Sequence = [1, 1/2, 2],
        scales: Sequence = [24, 48, 96, 192, 384],
        num_proto_masks: int = 32
    ) -> None:
        super().__init__(config)
        self.config = config
        self.num_classes = num_classes
        self.selected_layers = selected_layers
        self.fpn_channels = fpn_channels
        self.num_grids = num_grids
        self.mask_size = mask_size
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.num_proto_masks = num_proto_masks

        self.update_config(config)

        # TODO: rename backbone, neck, head layers
        self.backbone = resnet101()
        num_layers = max(self.selected_layers) + 1
        while len(self.backbone.layers) < num_layers:
            self.backbone.add_layer()

        self.freeze(self.training)

        self.config.mask_dim = config.mask_size**2

        self.neck = YolactPredictNeck(
            config, self.backbone.channels)
        # self.neck = YolactPredictNeck(
        #     config, [self.backbone.channels[i] for i in self.selected_layers])

        in_channels = self.fpn_channels
        in_channels += self.num_grids

        self.proto_layer = ProtoNet(in_channels, self.num_proto_masks)
        self.config.mask_dim = self.num_proto_masks

        # Preprocessing nested list
        self.aspect_ratios = [[self.aspect_ratios]] * len(self.neck.selected_layers)
        self.scales = [[scale] for scale in self.scales]
        self.heads = YolactPredictHead(
            config, self.neck.selected_layers, self.neck.channels,
            self.aspect_ratios, self.scales, self.num_classes
        )

        self.semantic_layer = SemanticSegmentation(
            self.neck.channels[0], self.num_classes-1, kernel_size=1
        )

        self.init_weights('cache/backbones/resnet50-19c8e357.pth')

    def init_weights(self, path):
        self.backbone.from_pretrained(path)

        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, inputs: List[Tensor], targets: Dict[str, Tensor] = None) -> Dict[str, List[Tensor]]:
        """
        Do not transform resized image, expected original image

        Args:
            inputs (:obj:`List[FloatTensor[B, C, H, W]]`):
            targets (:obj:):

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
                `size` (:obj:Tuple[int, int]):
        """
        inputs = self.check_inputs(inputs)
        # TODO: create resize modules for keep aspect ratio or min_size, h, w?
        inputs = F.interpolate(inputs, size=(self.config.max_size, self.config.max_size), mode='bilinear')
        self.config.device = inputs.device

        self.config.size = (inputs.size(2), inputs.size(3))

        outputs = self.backbone(inputs)
        # outputs = [outputs[i] for i in self.config.selected_layers]
        outputs = self.neck(outputs)

        return_dict = defaultdict(list)
        # TODO: self.neck.selected_layer...
        for i, head in zip(self.neck.selected_layers, self.heads):
            if head is not self.heads[0]:
                head.parent = [self.heads[0]]

            output = head(outputs[i])

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
            # TODO: ah si ba!! What should I do?!
            from .inference_yolact import YolactInference
            return_dict = YolactInference(81)(return_dict)
            return return_dict

    def load_weights(self, path):
        state_dict = torch.load(path)
        for key in list(state_dict.keys()):
            p = key.split('.')
            if p[0] == 'backbone':
                if p[1] == 'conv1':
                    new_key = f'backbone.conv.{p[2]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'bn1':
                    new_key = f'backbone.bn.{p[2]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[4].startswith('conv'):
                    new_key = f'backbone.layers.{p[2]}.{p[3]}.{p[4]}.0.{p[5]}'
                    state_dict[new_key] = state_dict.pop(key)
                else:
                    state_dict[key] = state_dict.pop(key)

            elif p[0] == 'fpn':
                if p[1] == 'lat_layers':
                    new_key = f'neck.lateral_layers.{p[2]}.{p[3]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'pred_layers':
                    new_key = f'neck.predict_layers.{p[2]}.{p[3]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'downsample_layers':
                    new_key = f'neck.extra_layers.{p[2]}.{p[3]}'
                    state_dict[new_key] = state_dict.pop(key)

            elif p[0] == 'prediction_layers':
                if p[2] == 'upfeature':
                    new_key = f'heads.0.upsample_layers.0.{p[4]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[2] == 'bbox_layer':
                    new_key = f'heads.0.box_layers.0.{p[3]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[2] == 'mask_layer':
                    new_key = f'heads.0.mask_layers.0.{p[3]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[2] == 'conf_layer':
                    new_key = f'heads.0.score_layers.0.{p[3]}'
                    state_dict[new_key] = state_dict.pop(key)

            elif p[0] == 'proto_net':
                new_key = f'proto_layer.{int(p[1])//2}.{p[2]}'
                state_dict[new_key] = state_dict.pop(key)

            elif p[0] == 'semantic_seg_conv':
                new_key = f'semantic_layer.0.{p[1]}'
                state_dict[new_key] = state_dict.pop(key)

        for key in list(state_dict.keys()):
            if key.startswith('neck.extra_layers.'):
                if int(key.split('.')[2]) >= 2:
                    state_dict.pop(key)

        self.load_state_dict(state_dict)
