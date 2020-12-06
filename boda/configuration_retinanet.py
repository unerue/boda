from typing import Tuple, List, Dict
from urllib.parse import MAX_CACHE_SIZE
from .configuration_base import BaseConfig


RETINANET_PRETRAINED_CONFIG = {
    'retinanet-base': None,
}


class RetinaNetConfig(BaseConfig):
    def __init__(
        self, 
        selected_layers=-1,
        grid_size=7, 
        num_boxes=2,
        max_size=448,
        num_classes=20,
        **kwargs):
        super().__init__(max_size=max_size, **kwargs)
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.selected_layers = list(range(1, 4))

 