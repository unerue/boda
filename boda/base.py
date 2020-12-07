import os
from typing import Tuple, List, Dict, Union

import torch
from torch import nn, Tensor


CONFIG_NAME = {}


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()


class Neck(nn.Module):
    def __init__(self):
        super().__init__()


class Head(nn.Module):
    def __init__(self):
        super().__init__()


class Model(nn.Module):
    """Base Model for 
    """
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
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike], *args, **kwargs):
        model_name_or_path = str(model_name_or_path)
        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, CONFIG_NAME)

    def load_weights(self, path):
        raise NotImplementedError

    def _check_pretrained_model_is_valid(self, model_name_or_path):
        # if model_name_or_path not in 
        pass
        
    @classmethod
    def get_config_dict(cls, model_name_or_path, **kwargs):
        pass


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
                elif target['labels'].dim () != 1:
                    raise ValueError('Expected target boxes to be a tensor of [N].')
            else:
                raise ValueError('Expected target boxes to be Tensor.')
            break
