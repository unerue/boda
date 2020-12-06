from typing import Tuple, List, Dict
from urllib.parse import MAX_CACHE_SIZE
from .configuration_base import BaseConfig


FCN_PRETRAINED_CONFIG = {
    'fcn8s': None,
    'fcn16s': None
}


class FcnConfig(BaseConfig):
    """Configuration for FCN"""
    def __init__(
        self, 
        selected_layers=-1,
        grid_size=7, 
        num_boxes=2,
        max_size=448,
        num_classes=20,
        **kwargs):
        super().__init__(max_size=max_size, **kwargs)
        self.selected_layers = selected_layers
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        
        
