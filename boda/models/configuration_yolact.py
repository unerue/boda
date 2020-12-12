import os
from typing import Union, Any
from ..configuration_base import BaseConfig


YOLACT_PRETRAINED_CONFIG = {
    'yolact550-base': None,
    'yolact300': None,
    'yolact700': None,
}


class YolactConfig(BaseConfig):
    """Configuration for YOLACT

    Arguments:
        max_size ():
        padding ():
        proto_net_structure (List):
    """
    config_name = 'yolact550-base'

    def __init__(
        self,
        max_size=550,
        padding=1,
        use_conv_downsample=True,
        num_features=256,
        num_grids=0,
        mask_size=16,
        mask_dim=0,
        proto_net_structure=None,
        head_layer_params=None,
        extra_layers=(0, 0, 0),
        extra_net_structure=None,
        **kwargs
    ) -> None:
        super().__init__(max_size=max_size, **kwargs)
        # self.selected_layers = list(range(1, 4))
        # self.num_boxes = None
        self.num_classes = 7
        # neck
        self.padding = 1
        self.aspect_ratios = [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]]*6
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
        self.proto_net = [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
        self.mask_proto_prototypes_as_features = None
        self.mask_proto_prototypes_as_features_no_grad = None
        self.mask_proto_coef_activation = None
        self.mask_proto_normalize_emulate_roi_pooling = True
        self.mask_proto_bias = None
        self.head_layer_params = {'kernel_size': 3, 'padding': 1}
        self.extra_head_net = [(256, 3, {'padding': 1})]
        self.extra_layers = (0, 0, 0)
        self.eval_mask_branch = True

        self.num_extra_bbox_layers = 0
        self.num_extra_conf_layers = 0
        self.num_extra_mask_layers = 0
        self.freeze_bn = True

        self.preapply_sqrt = True
        self.use_pixel_scales = True
        self.use_square_anchors = True