from ..configuration_base import BaseConfig


YOLOV1_PRETRAINED_CONFIG = {
    'yolov1-base': 'https://drive.google.com/file/d/10cpkJnhDLZr-Vtt8zNTf-8kz9gl3S0He/view?usp=sharing',
    'yolov1-tiny': '',
}


class Yolov1Config(BaseConfig):
    """Configuration for YOLOv1

    Arguments:
        max_size (Union[int, Tuple[int]]):
        num_classes (int):
        selected_layers (Union[int, List[int]]):
    """
    model_name = 'yolov1'

    def __init__(
        self,
        num_classes=20,
        max_size=448,
        backbone_name=None,
        backbone_structure=None,
        selected_layers=-1,
        num_boxes=2,
        num_grids=7,
        bn=False,
        relu=False,
        **kwargs
    ) -> None:
        super().__init__(max_size=max_size, **kwargs)
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.backbone_structure = backbone_structure
        self.selected_layers = selected_layers
        self.num_boxes = num_boxes
        self.num_grids = num_grids
        # TODO: rename or arange backbone config
        self.bn = bn
        self.relu = relu
        # TODO: rename and arange loss function config
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.class_scale = 1
        self.coord_scale = 5
        self.jitter = 0.2
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.sqrt = True
