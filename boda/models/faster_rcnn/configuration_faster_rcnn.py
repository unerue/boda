import os
from typing import Union, Any, Sequence, List
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
        num_classes: int = 80,
        min_size: int = 800,
        max_size: int = 1333,
        preserve_aspect_ratio: bool = True,

        fnp_channels: int = 256,
        num_extra_fpn_layers: int = 1,

        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,

        rpn_nms_threshold: float = 0.7,
        rpn_fg_iou_threshold: float = 0.7,
        rpn_bg_iou_threshold: float = 0.3,
        rpn_batch_size_per_image: float = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_score_threshold: float = 0.0,

        aspect_ratios: List[float] = [0.5, 1.0, 2.0],
        anchors: List[int] = (32, 64, 128, 256, 512),

        box_score_threshold: float = 0.05,
        box_nms_threshold: float = 0.5,
        box_detections_per_image: int = 100,
        box_fg_iou_threshold: float = 0.5,
        box_bg_iou_threshold: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        bbox_reg_weights: float = None,

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
        super().__init__(**kwargs)
        self.num_classes = num_classes + 1
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios
        self.anchors = anchors

        self.rpn_pre_nms_top_n_train = rpn_pre_nms_top_n_train
        self.rpn_pre_nms_top_n_test = rpn_pre_nms_top_n_test
        self.rpn_post_nms_top_n_train = rpn_post_nms_top_n_train
        self.rpn_post_nms_top_n_test = rpn_post_nms_top_n_test
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_fg_iou_threshold = rpn_fg_iou_threshold
        self.rpn_bg_iou_threshold = rpn_bg_iou_threshold
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.rpn_positive_fraction = rpn_positive_fraction
        self.rpn_score_threshold = rpn_score_threshold

        self.box_score_threshold = box_score_threshold
        self.box_nms_threshold = box_nms_threshold
        self.box_detections_per_image = box_detections_per_image
        self.box_fg_iou_threshold = box_fg_iou_threshold
        self.box_bg_iou_threshold = box_bg_iou_threshold
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction
        self.bbox_reg_weights = bbox_reg_weights

        
