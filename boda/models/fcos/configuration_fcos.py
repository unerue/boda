import os
from typing import Union, Any, Sequence
from ...base_configuration import BaseConfig


class FcosConfig(BaseConfig):
    """Configuration for FCOS

    Args:
        max_size ():
        padding ():
        proto_net_structure (List):
    """
    config_name = 'fcos'

    def __init__(
        self,
        min_size: int = 800,
        max_size: int = 1333,
        preserve_aspect_ratio: bool = True,
        num_classes: int = 80,
        fnp_channels: int = 256,
        num_extra_fpn_layers: int = 1,
        num_box_layers: int = 4,
        num_score_layers: int = 4,
        num_share_layers: int = 0,
        strides: Sequence[int] = [8, 16, 32, 64, 128],
        gamma: float = 2.0,
        alpha: float = 0.25,
        box_weight: float = 1.0,
        score_weight: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(max_size=max_size, **kwargs)
        self.num_classes = num_classes + 1
