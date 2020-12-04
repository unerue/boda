from typing import Tuple, List, Dict
from urllib.parse import MAX_CACHE_SIZE
from .configuration_base import PretrainedConfig


YOLOV1_PRETRAINED_CONFIG = {
    'yolact-base': None,
    'yolact-300': None,
    'yolact-700': None,
}


class YolactConfig(PretrainedConfig):
    def __init__(
        self, *args,
        selected_layers=-1,
        max_size=448,
        num_classes=20,
        **kwargs):
        super().__init__(max_size=max_size, **kwargs)
        self.backbone = 'resnet101'
        self.selected_layers = list(range(1, 4))
        self.num_boxes = None
        self.num_classes = num_classes
        # neck
        self.aspect_ratios = [[[1/2, 1, 2]]] * 5
        self.pred_scales = [[24], [48], [96], [192], [384]]
        self.num_features = 256
        self.interpolation_mode = 'bilinear'
        self.num_downsample = 1
        self.use_conv_downsample = True
        self.padding = True
        self.relu_downsample_layers = False
        self.relu_pred_layers = True
        # head
        self.mask_type = 1
        self.mask_alpha = 6.125
        self.mask_proto_src = 0
        self.mask_proto_net = [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
        self.mask_proto_prototypes_as_features = None
        self.mask_proto_prototypes_as_features_no_grad = None
        self.mask_proto_coef_activation = None
        self.mask_proto_normalize_emulate_roi_pooling = True
        self.mask_proto_bias = None
        self.head_layer_params = {'kernel_size': 3, 'padding': 1}
        self.extra_head_net = [(256, 3, {'padding': 1})]
        self.extra_layers = (0, 0, 0)
        self.eval_mask_branch = True