from ..config import BaseConfig


YOLOV2_PRETRAINED_CONFIG = {
    'yolov2-base': None,
    'yolov2-tiny': None,
}


class Yolov2Config(BaseConfig):
    model_name = 'yolov2'

    def __init__(
        self,
        selected_layers=-1,
        num_grids=7,
        num_boxes=2,
        max_size=448,
        num_classes=20,
        **kwargs
    ) -> None:
        super().__init__(max_size=max_size, **kwargs)
        self.selected_layers = selected_layers
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.class_scale = 1
        self.coord_scale = 5
        self.jitter = 0.2
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.sqrt = 1
