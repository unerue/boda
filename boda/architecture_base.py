import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Union

import torch
from torch import nn, Tensor


class Backbone(nn.Module):
    backbone_name: str = ''

    def __init__(self):
        super().__init__()
        self.channels: List[int] = []

    def forward(self, inputs: Tensor) -> List[Tensor]:
        raise NotImplementedError

    @torch.jit.unused
    def eager_outputs(self, *args):
        raise NotImplementedError

    def init_weights(self, *args):
        raise NotImplementedError

    def from_pretrained(self, backbone_name, **kwargs):
        from torch.hub import load_state_dict_from_url

        # state_dict = load_state_dict_from_url(BACKBONE_ARCHIVE_MAP[backbone_name])
        self.load_state_dict(state_dict)

    def _from_state_dict(self, *args):
        raise NotImplementedError


class Neck(nn.Module):
    neck_type: str = ''

    def __init__(self):
        super().__init__()
        self.channels: List[int] = []

    def _add_extra_layer(self):
        raise NotImplementedError

    def eager_outputs(self, *args):
        raise NotImplementedError

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels: List[int] = []

    def _add_predict_layer(self):
        raise NotImplementedError

    def eager_outputs(self, *args):
        raise NotImplementedError

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError


class ModelMixin(ABC):
    """Base Model for Computer Vision Models
    """
    model_name: str = ''
    _checked_inputs: bool = True

    def __init__(self, config, **kwargs):
        ...

    def initialize(self, config, backbone, neck, head):
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls):
        """Create from pretrained model weights """
        ...

    @classmethod
    def check_inputs(cls, inputs):
        """
        Argument:
            inputs (List[Tensor]): Size([C, H, W])
        Return:
            outputs (Tensor): Size([B, C, H, W])
        """
        if cls._checked_inputs:
            print('Check!!')
            for image in inputs:
                if isinstance(image, Tensor):
                    if image.dim() != 3:
                        raise ValueError(f'images is expected to be 3d tensors of shape [C, H, W] {image.size()}')
                else:
                    raise ValueError('Expected image to be Tensor.')
            cls._checked_inputs = False

        if isinstance(inputs, list):
            inputs = torch.stack(inputs)

        return inputs


class Model(nn.Module, ModelMixin):
    config_class = None
    base_model_prefix: str = ''

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config
        self.name_or_path = ''

    # @property
    # def base_model(self) -> nn.Module:
    #     return getattr(self, self.base_model_prefix, self)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike],
        **kwargs
    ):
        raise NotImplementedError

    # def load_weights(self, path):
    #     raise NotImplementedError

    # def _check_pretrained_model_is_valid(self, model_name_or_path):
    #     # if model_name_or_path not in
    #     raise NotImplementedError

    # @classmethod
    # def get_config_dict(cls, model_name_or_path, **kwargs):
    #     raise NotImplementedError


class Matcher(ABC):
    def __init__(self):
        pass

    def encode(self):
        pass

    def deconde(self):
        pass


class LossFunction(nn.Module):
    _checked_targets = True

    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        inputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    @classmethod
    def copy_targets(cls, targets: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        if targets is not None:
            targets_copy: List[Dict[str, Tensor]] = []
            for target in targets:
                _target: Dict[str, Tensor] = {}
                for key, value in target.items():
                    _target[key] = value
                targets_copy.append(_target)
            targets = targets_copy

        return targets

    @classmethod
    def check_targets(cls, targets: List[Dict[str, Tensor]]) -> None:
        if cls._checked_targets:
            for target in targets:
                if isinstance(target['boxes'], Tensor):
                    boxes = target['boxes']
                    check_boxes = boxes[:, :2] >= boxes[:, 2:]
                    if boxes.dim() != 2 or boxes.size(1) != 4:
                        raise ValueError('Expected target boxes to be a tensor of [N, 4].')
                    elif check_boxes.any():
                        raise ValueError(f'{boxes}')
                    elif target['labels'].dim() != 1:
                        raise ValueError('Expected target boxes to be a tensor of [N].')
                else:
                    raise ValueError('Expected target boxes to be Tensor.')
            cls._checked_targets = False
    
    def decode(self, targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        for target in targets:
            pass


# class Register(ABCMeta):
#     registry = {}
#     def __new__(cls):
#         new_cls = type.__new__(cls, name, bases, attrs)
#         if not hasattr(new_cls, '_registry_name'):
#             raise Exception('Ay class')

#         cls.register[new_cls._registry_name] = new_cls
#         return ABCMeta.__new__(cls, name, bases, attrs)

#     @classmethod
#     def get_registry(cls):
#         return dict(cls.registry)

    # @classmethod
    # def __subclasshook__(cls, subclass):
    #     return hasattr(subclass, 'from_pretrained') or NotImplementedError


# registry = []

# def register(func):
#     print(f'running register {func}')
#     registry.append(func.__class__.__name__)
#     return func
