import os
from typing import Optional, Tuple, Union, Any
from ..configuration_base import BaseConfig


class YolactEdgeConfig(BaseConfig):
    """Configuration for YOLACT

    Args:
        max_size (:obj:`Union[int, Tuple[int]]`):
        num_classes (:obj:`int`):
        num_grids (:obj:`int`):
        num_grid_sizes (:obj:`int`):
        num_mask_dim (:obj:`int`):

        padding (:obj:`int`):
        use_conv_downsample (:obj:`bool`, defaults to `True`): 
        extra_layers (:obj:`int`): 
        extra_layer_structure (:obj:`int`):
        proto_layer_structure (:obj:`List[]):
        head_layer_params (:obj:`List`)
        mask_dim (:obj:`int`): 
        num_grid_sizes (:obj:`int`):
        num_mask_dim (:obj:`int`):

    """
    model_name = 'yolact'

    def __init__(
        self,
        num_classes: int = 80,
        max_size: int = 550,
        padding: int = 1,
        use_conv_downsample: bool = True,
        num_features: int = 256,
        num_grids: int = 0,
        mask_size: int = 16,
        mask_dim: int = 0,
        proto_net_structure: Optional[int] = None,
        head_layer_params=None,
        extra_layers: Tuple[int] = (0, 0, 0),
        extra_net_structure=None,
        **kwargs
    ) -> None:
        super().__init__(max_size=max_size, **kwargs)
        # self.selected_layers = list(range(1, 4))
        # self.num_boxes = None
        self.num_classes = num_classes + 1
        # neck
        self.padding = 1
        self.aspect_ratios = [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]] * 6
        # self.pred_scales = [[24], [48], [96], [192], [384]]
        # self.fpn_out_channels = 256
        # self.predict_channels = 256
        # self.interpolate_mode = 'bilinear'
        self.num_downsamples = 2
        self.use_conv_downsample = True
        # self.padding = True
        self.padding = 1
        self.relu_downsample_layers = False
        self.relu_pred_layers = True
        # head
        self.num_grids = 0
        self.mask_size = 16
        self.mask_dim = None
        self.mask_type = 1
        self.mask_alpha = 6.125
        self.proto_src = 0
        self.proto_net = [(256, 3, {'padding': 1})]*3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
        self.mask_proto_prototypes_as_features = None
        self.mask_proto_prototypes_as_features_no_grad = None
        self.mask_proto_coef_activation = None
        self.mask_proto_normalize_emulate_roi_pooling = True
        self.mask_proto_bias = None
        self.head_layer_params = {'kernel_size': 3, 'padding': 1}
        self.extra_head_net = [(256, 3, {'padding': 1})]
        self.extra_layers = (0, 0, 0)
        self.eval_mask_branch = True
        self.use_share = False
        self.use_semantic_segmentation = True
        self.use_eval_mask_branch = True

        self.num_extra_bbox_layers = 0
        self.num_extra_conf_layers = 0
        self.num_extra_mask_layers = 0
        self.freeze_bn = True

        self.use_preapply_sqrt = True
        self.use_pixel_scales = True
        self.use_square_anchors = True

        self.label_map = {}