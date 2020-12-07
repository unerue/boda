import os
from typing import Tuple, List, Dict, Union

import torch
from torch import nn, Tensor


BACKBONE_ARCHIVE_MAP = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class Backbone(nn.Module):
    backbone_name: str = ''

    def __init__(self):
        super().__init__()

    @torch.jit.unused
    def eager_outputs(self, *args):
        raise NotImplementedError

    def init_weights(self, *args):
        raise NotImplementedError

    def from_pretrained(self, backbone_name, **kwargs):
        from torch.hub import load_state_dict_from_url

        state_dict = load_state_dict_from_url(BACKBONE_ARCHIVE_MAP[backbone_name])
        self.load_state_dict(state_dict)

    def _from_state_dict(self, *args):
        raise NotImplementedError


class Neck(nn.Module):
    def __init__(self):
        super().__init__()

    def eager_outputs(self, *args):
        raise NotImplementedError


class Head(nn.Module):
    def __init__(self):
        super().__init__()

    def eager_outputs(self, *args):
        raise NotImplementedError


class Model(nn.Module):
    """Base Model for
    """
    model_name: str = ''
    checked_inputs = True

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def check_inputs(cls, inputs):
        """
        Argument:
            inputs (List[Tensor]): Size([C, H, W])
        Return:
            outputs (Tensor): Size([B, C, H, W])
        """
        if cls.checked_inputs:
            print('Check!!')
            for image in inputs:
                if isinstance(image, Tensor):
                    if image.dim() != 3:
                        raise ValueError(f'images is expected to be 3d tensors of shape [C, H, W] {image.size()}')
                else:
                    raise ValueError('Expected image to be Tensor.')
            cls.checked_inputs = False

        if isinstance(inputs, list):
            inputs = torch.stack(inputs)

        return inputs

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike], **kwargs):
        raise NotImplementedError

    def load_weights(self, path):
        raise NotImplementedError

    def _check_pretrained_model_is_valid(self, model_name_or_path):
        # if model_name_or_path not in 
        raise NotImplementedError

    @classmethod
    def get_config_dict(cls, model_name_or_path, **kwargs):
        raise NotImplementedError


class LoseFunction(nn.Module):
    checked_targets = True

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def copy_targets(cls, targets):
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
    def check_targets(cls, targets: List[Dict[str, Tensor]]):
        for target in targets:
            if isinstance(target['boxes'], Tensor):
                if target['boxes'].dim() != 2 or target['boxes'].size(1) != 4:
                    raise ValueError('Expected target boxes to be a tensor of [N, 4].')
                elif target['labels'].dim() != 1:
                    raise ValueError('Expected target boxes to be a tensor of [N].')
            else:
                raise ValueError('Expected target boxes to be Tensor.')
            break
