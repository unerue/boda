import os
from typing import Union, Any
from ...base_configuration import BaseConfig


class CenterMaskConfig(BaseConfig):
    """Configuration for CenterMaskv1

    Arguments:
        max_size ():
        padding ():
        proto_net_structure (List):
    """
    config_name = 'centermask'

    def __init__(
        self,
        max_size=550,
        num_classes=80,
        **kwargs
    ) -> None:
        super().__init__(max_size=max_size, **kwargs)
        self.num_classes = num_classes + 1
