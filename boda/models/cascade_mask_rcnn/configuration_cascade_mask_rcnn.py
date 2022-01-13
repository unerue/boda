from typing import List, Tuple

from ...base_configuration import BaseConfig


class CascadeMaskRCNNConfig(BaseConfig):
    model_name = 'cascade_mask_rcnn'
    
    def __init__(
        self,
        min_size=800,
        max_size=1333,
        num_classes: int = 80,
        preserve_aspect_ratio: bool = True,
        backbone_name: str = 'resnet50',
        selected_backbone_layers: List[int] = [0, 1, 2, 3],
        neck_name: str = 'fpn',
        fpn_channels: int = 256,
        fpn_num_extra_predict_layers: int = 0,
        anchor_sizes: Tuple[int] = (32, 64, 128, 256, 512),
        aspect_ratios: Tuple[int] = (0.5, 1.0, 2.0),
        rpn_box_coder_weights: Tuple = (1.0, 1.0, 1.0, 1.0),
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_score_thresh: float = 0.0,
        box_roi_pool_feat_names: List[str] = ['0', '1', '2', '3'],
        box_roi_pool_out_size: int = 7,
        box_roi_pool_sample_ratio: int = 0,
        representation_size: int = 1024,
        box_roi_reg_class_agnostic: bool = True,
        box_score_thresh: float = 0.05,
        box_detections_per_img: int = 100,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        box_nms_thresh_1: float = 0.5,
        box_fg_iou_thresh_1: float = 0.5,
        box_bg_iou_thresh_1: float = 0.5,
        bbox_reg_weights_1: List[float] = [10., 10., 5., 5.],
        box_nms_thresh_2: float = 0.6,
        box_fg_iou_thresh_2: float = 0.6,
        box_bg_iou_thresh_2: float = 0.6,
        bbox_reg_weights_2: List[float] = [20., 20., 10., 10.],
        box_nms_thresh_3: float = 0.7,
        box_fg_iou_thresh_3: float = 0.7,
        box_bg_iou_thresh_3: float = 0.7,
        bbox_reg_weights_3: List[float] = [30., 30., 15., 15.],
        mask_roi_pool_feat_names: List[str] = ['0', '1', '2', '3'],
        mask_roi_pool_out_size: int = 14,
        mask_roi_pool_sample_ratio: int = 0,
        mask_head_in_channels: int = 256,
        mask_layers: Tuple[int] = (256, 256, 256, 256),
        mask_dilation: int = 1,
        mask_predictor_in_channels: int = 256,
        mask_dim_reduced: int = 256,
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes+1,
            min_size=min_size,
            max_size=max_size,
            backbone_name=backbone_name,
            neck_name=neck_name,
            fpn_channels=fpn_channels,
            preserve_aspect_ratio=preserve_aspect_ratio,
            **kwargs
        )

        self.selected_backbone_layers = selected_backbone_layers
        self.fpn_num_extra_predict_layers = fpn_num_extra_predict_layers
        
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.rpn_box_coder_weights = rpn_box_coder_weights
        self.rpn_pre_nms_top_n_train = rpn_pre_nms_top_n_train
        self.rpn_pre_nms_top_n_test = rpn_pre_nms_top_n_test
        self.rpn_post_nms_top_n_train = rpn_post_nms_top_n_train
        self.rpn_post_nms_top_n_test = rpn_post_nms_top_n_test
        self.rpn_nms_thresh = rpn_nms_thresh
        self.rpn_fg_iou_thresh = rpn_fg_iou_thresh
        self.rpn_bg_iou_thresh = rpn_bg_iou_thresh
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.rpn_positive_fraction = rpn_positive_fraction
        self.rpn_score_thresh = rpn_score_thresh
        
        self.box_roi_pool_feat_names = box_roi_pool_feat_names
        self.box_roi_pool_out_size = box_roi_pool_out_size
        self.box_roi_pool_sample_ratio = box_roi_pool_sample_ratio
        self.representation_size = representation_size
        self.box_roi_reg_class_agnostic = box_roi_reg_class_agnostic
        
        self.box_score_thresh = box_score_thresh
        self.box_detections_per_img = box_detections_per_img
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction
        self.box_nms_thresh_1 = box_nms_thresh_1
        self.box_fg_iou_thresh_1 = box_fg_iou_thresh_1
        self.box_bg_iou_thresh_1 = box_bg_iou_thresh_1
        self.bbox_reg_weights_1 = bbox_reg_weights_1
        self.box_nms_thresh_2 = box_nms_thresh_2
        self.box_fg_iou_thresh_2 = box_fg_iou_thresh_2
        self.box_bg_iou_thresh_2 = box_bg_iou_thresh_2
        self.bbox_reg_weights_2 = bbox_reg_weights_2
        self.box_nms_thresh_3 = box_nms_thresh_3
        self.box_fg_iou_thresh_3 = box_fg_iou_thresh_3
        self.box_bg_iou_thresh_3 = box_bg_iou_thresh_3
        self.bbox_reg_weights_3 = bbox_reg_weights_3
        
        self.mask_roi_pool_feat_names = mask_roi_pool_feat_names
        self.mask_roi_pool_out_size = mask_roi_pool_out_size
        self.mask_roi_pool_sample_ratio = mask_roi_pool_sample_ratio
        self.mask_head_in_channels = mask_head_in_channels
        self.mask_layers = mask_layers
        self.mask_dilation = mask_dilation
        self.mask_predictor_in_channels = mask_predictor_in_channels
        self.mask_dim_reduced = mask_dim_reduced
