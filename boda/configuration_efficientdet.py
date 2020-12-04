from typing import Tuple, List, Dict
from urllib.parse import MAX_CACHE_SIZE
from .configuration_base import PretrainedConfig


YOLOV1_PRETRAINED_CONFIG = {
    'efficientdet-d0': None,
    'efficientdet-d1': None,
    'efficientdet-d2': None,
    'efficientdet-d3': None,
}


class EfficientDetConfig(PretrainedConfig):
    def __init__(
        self, 
        selected_layers=-1,
        grid_size=7, 
        num_boxes=2,
        max_size=512,
        num_classes=20,
        **kwargs):
        super().__init__(max_size=max_size, **kwargs)
        self.selected_layers = selected_layers
        self.num_classes = num_classes

        self.fpn_channels = 64
        self.fpn_cell_repeats = 3
        self.box_class_repeats = 3
        self.pad_type = ''
        self.redundant_bias = False

        # feature + anchor
        self.min_level = 3
        self.max_level = 7
        self.num_levles = self.min_level + self.max_level
        self.num_scales = 3
        self.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.anchor_scale = 4.0

        # FPN
        self.pad_type = 'same'
        self.act_type = 'swish'
        self.norm_layer = None
        self.norm_kwargs = dict(eps=.001, momentum=.01)
        self.box_class_repeats = 3
        self.fpn_cell_repeats = 3
        self.fpn_channels = 88
        self.separable_conv = True
        self.apply_bn_for_resampling = True
        self.conv_after_downsample = False
        self.conv_bn_relu_pattern = False
        self.use_native_resize_op = False

        # classification loss
        self.alpha = 0.25
        self.gamma = 1.5
        self.label_smoothing = 0.and
        self.new_focal = False
        self.jit_loss = False

        # localization loss
        self.delta = 0.1
        self.box_loss_weight = 50.0



