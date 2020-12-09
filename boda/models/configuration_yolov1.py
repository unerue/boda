from ..configuration_base import BaseConfig


YOLOV1_PRETRAINED_CONFIG = {
    'yolov1-base': None,
    'yolov1-tiny': None,
}


class Yolov1Config(BaseConfig):
    """Configuration for YOLOv1

    Arguments:
        max_size ():
        num_classes ():
        backbone_structure ():
        seletected_layers ():
        num_boxes ():
        grid_size ():
    """
    model_name = 'yolov1'
    def __init__(
        self, 
        max_size=448,
        num_classes=20,
        backbone_structure=None,
        selected_layers=[-1],
        num_boxes=2,
        grid_size=7,
        **kwargs):
        super().__init__(max_size=max_size, **kwargs)
        self.num_classes = num_classes
        self.backbone_structure = backbone_structure
        self.selected_layers = selected_layers
        self.num_boxes = num_boxes
        self.num_grids = 7
        self.grid_size = grid_size

        self.bn = True
        self.relu = False

        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.class_scale = 1
        self.coord_scale = 5
        self.jitter = 0.2
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.sqrt = True
