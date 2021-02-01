from typing import Union

import torch
from torch import nn, Tensor
from ...base_architecture import Neck, Head, Model


class MaskRcnnNeck(Neck):
    def __init__(self):
        ...


class MaskRcnnHead(Head):
    def __init__(self):
        ...


class MaskRcnnPretrained(Model):
    config_class = MaskRcnnConfig
    base_model_prefix = 'mask_rcnn'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = MaskRcnnModel(config)
        # model.state_dict(torch.load('test.pth'))
        return model


class MaskRcnnModel(MaskRcnnPretrained):
    """Mask R-CNN
    """
    model_name = 'mask_rcnn'

    def __init__(self):
        ...






