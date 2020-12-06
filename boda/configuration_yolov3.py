from .configuration_base import BaseConfig


YOLOV3_PRETRAINED_CONFIG = {
    'yolov3-base': None,
    'yolov3-tiny': None,
}


class Yolov3Config(BaseConfig):
    model_name = 'yolov3'
    def __init__(
        self, 
        selected_layers=-1,
        grid_size=7, 
        num_boxes=2,
        max_size=416,
        num_classes=20,
        **kwargs):
        super().__init__(max_size=max_size, **kwargs)
        self.selected_layers = selected_layers
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.class_scale = 1
        self.coord_scale = 5
        self.jitter = 0.3
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.sqrt = 1
        self.anchors = ((10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326))
        self.ignore_thresh = 0.7
        self.truth_thresh = 1.0