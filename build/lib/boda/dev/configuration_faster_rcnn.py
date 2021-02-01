from typing import Tuple, List, Dict
from urllib.parse import MAX_CACHE_SIZE
from .configuration_base import BaseConfig


FASTER_RCNN_PRETRAINED_CONFIG = {
    'faster-rcnn-base': None,
}


class FasterRcnnConfig(BaseConfig):
    """Configuration for Faster R-CNN"""
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

        self.min_size = 800
        self.max_size = 1333
        self.rpn_pre_nms_top_n_train = 2000
        self.rpn_pre_nms_top_n_test = 1000
        self.rpn_post_nms_top_n_train = 2000
        self.rpn_post_nms_top_n_test = 1000
        self.rpn_nms_thresh = 0.7
        self.rpn_fg_iou_thresh = 0.7
        self.rpn_bg_iou_thresh = 0.3
        self.rpn_batch_size_per_image = 256
        self.rpn_positive_fraction = 0.5

        self.box_score_thresh = 0.05
        self.box_nms_thresh = 0.5
        self.box_detections_per_image = 100
        self.box_fg_iou_thresh = 0.05
        self.box_bg_iou_thresh = 0.5
        self.box_batch_size_per_image = 512
        self.box_positive_fraction = 0.25
        self.box_reg_weights = None

        self.anchor_sizes = ((32,), (64,), (128,), (256,), (512))
        self.aspect_ratios = ((0.5, 1.0, 2.0)) * len(self.anchor_sizes)
        