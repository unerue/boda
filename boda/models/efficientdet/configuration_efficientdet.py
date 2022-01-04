from ...base_configuration import BaseConfig


class EfficientDetConfig(BaseConfig):
    model_name = 'efficientdet'
    
    def __init__(
        self,
        num_classes: int = 90,  # rwightman
        aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        **kwargs,
    ):
        # !TODO: add arch name exception
        # self.compound_coef = 0
        # self.compound_coef = 1
        self.compound_coef = 7
                
        # self.fpn_num_filter = 64
        # self.fpn_num_filter = 88
        self.fpn_num_filter = 384
        # self.fpn_cell_repeat = 3
        # self.fpn_cell_repeat = 4
        self.fpn_cell_repeat = 8
        # self.input_size = 512
        # self.input_size = 640
        self.input_size = 1536
        # self.box_class_repeat = 3
        # self.box_class_repeat = 3
        self.box_class_repeat = 5
        # self.pyramid_level = 5
        # self.pyramid_level = 5
        self.pyramid_level = 6
        # self.anchor_scale = 4.0
        # self.anchor_scale = 4.0
        self.anchor_scale = 4.0
        self.num_scales = len(scales)
        # self.conv_channel_coef = [40, 112, 320]
        # self.conv_channel_coef = [40, 112, 320]
        self.conv_channel_coef = [80, 224, 640]

        self.num_anchors = len(aspect_ratios) * self.num_scales
        
        super().__init__(
            num_classes=num_classes,
            # backbone_name='efficientnet_b0',
            # backbone_name='efficientnet_b1',
            backbone_name='efficientnet_b7',
            neck_name='bifpn',
            aspect_ratios=aspect_ratios,
            scales=scales,
            fpn_channels=self.fpn_num_filter,
            **kwargs,
        )
