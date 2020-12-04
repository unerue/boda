from typing import Tuple, List, Dict
from urllib.parse import MAX_CACHE_SIZE
from .configuration_base import PretrainedConfig


YOLOV1_PRETRAINED_CONFIG = {
    'yolov1-base': None,
    'yolov1-tiny': None,
}


class SsdConfig(PretrainedConfig):
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
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        # self.max_size = max_size
        self.num_classes = num_classes
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.class_scale = 1
        self.coord_scale = 5
        self.jitter = 0.2
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.sqrt = 1

