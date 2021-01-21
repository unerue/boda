from typing import Union

import torch
from torch import nn, Tensor
from ...base_architecture import Neck, Head, Model


class CenterMaskNeck(Neck):
    def __init__(self):
        ...


class CenterMaskHead(Head):
    def __init__(self):
        ...


class CenterMaskPretrained(Model):
    config_class = CenterMaskConfig
    base_model_prefix = 'centermask'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = CenterMaskModel(config)
        # model.state_dict(torch.load('test.pth'))
        return model


class CenterMaskModel(CenterMaskPretrained):
    """
    """
    model_name = 'centermaskv1'

    def __init__(self):
        ...



