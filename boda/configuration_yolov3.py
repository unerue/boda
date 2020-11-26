from .configuration_base import PretrainedConfig


YOLOV3_PRETRAINED_CONFIG = {
    'yolov3-base': 'https://unerue.synology.me:5001&'
    'yolov3-': None,
}


class Yolov3Config(PretrainedConfig):
    model = 'yolov3'
    def __init__(
        self, 
        mask=[[1, 2, 3], [4, 5, 6], [6, 7, 8]], 
        anchors=[[10, 13], [16, 30], [33,23], [30, 61], [62, 45], [59, 119], [156, 198], [373, 326]],
        ignore_thresh=0.7,
        truth_thresh=1,
        random=1,
        num_classes=80,
        obj_scale=1,
        noobj_scale=100,
        grid_size=0,
        
        ):
