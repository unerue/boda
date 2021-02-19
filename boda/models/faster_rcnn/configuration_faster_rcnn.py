import os
from typing import Union, Any, Sequence
from ...base_configuration import BaseConfig


class FasterRcnnConfig(BaseConfig):
    """Configuration for Faster R-CNN

    Args:
        max_size ():
        padding ():
        proto_net_structure (List):
    """
    config_name = 'fcos'

    def __init__(
        self,
        max_size: Sequence[int] = (1333, 800),
        num_classes: int = 80,
        fnp_channels: int = 256,
        num_extra_fpn_layers: int = 1,
        num_box_layers: int = 4,
        num_score_layers: int = 4,
        num_share_layers: int = 0,
        strides: Sequence[int] = [8, 16, 32, 64, 128],
        gamma: float = 2.0,
        alpha: float = 0.25,
        box_weight: float = 1.0,
        score_weight: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(max_size=max_size, **kwargs)
        self.num_classes = num_classes + 1
        self.min_size = 800,
        self.max_size = 1333,
        self.rpn_anchor_generator = None,
        self.rpn_head = None,
        self.rpn_pre_nms_top_n_train = 2000,
        self.rpn_pre_nms_top_n_test = 1000,
        self.rpn_post_nms_top_n_train = 2000,
        self.rpn_post_nms_top_n_test = 1000,
        self.rpn_nms_thresh = 0.7,
        self.rpn_fg_iou_thresh = 0.7,
        self.rpn_bg_iou_thresh = 0.3,
        self.rpn_batch_size_per_image = 256,
        self.rpn_positive_fraction = 0.5,
        self.rpn_score_thresh = 0.0,
        # Box parameters
        self.box_roi_pool = None,
        self.box_head = None,
        self.box_predictor = None,
        self.box_score_thresh = 0.05,
        self.box_nms_thresh = 0.5,
        self.box_detections_per_img = 100,
        self.box_fg_iou_thresh = 0.5,
        self.box_bg_iou_thresh = 0.5,
        self.box_batch_size_per_image = 512,
        self.box_positive_fraction = 0.25,
        self.bbox_reg_weights = None,

        self.aspect_ratios = (0.5, 1.0, 2.0)
        self.anchors = (32, 64, 128, 256, 512)
