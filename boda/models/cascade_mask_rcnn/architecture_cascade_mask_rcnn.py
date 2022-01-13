import os
from typing import Union, Sequence
from collections import OrderedDict

import torch
from torch import nn
from torchvision._internally_replaced_utils import load_state_dict_from_url

from ...base_architecture import Model
from .configuration_cascade_mask_rcnn import CascadeMaskRCNNConfig
from ..backbone_resnet import resnet18, resnet34, resnet50, resnet101
from ..neck_fpn import FeaturePyramidNetworks
from ..faster_rcnn.rpn import RegionProposalNetwork, RpnHead
from ..faster_rcnn.anchor_generator import AnchorGenerator
from ..faster_rcnn.roi_heads import MultiScaleRoIAlign, RoiHeads
from ..faster_rcnn.architecture_faster_rcnn import LinearHead


class CascadeMaskRCNNNeck(FeaturePyramidNetworks):
    def __init__(
        self,
        config: CascadeMaskRCNNConfig,
        channels: Sequence[int],
    ) -> None:
        super().__init__(
            channels=channels,
            selected_layers=config.selected_backbone_layers,
            out_channels=config.fpn_channels,
            num_extra_predict_layers=config.fpn_num_extra_predict_layers,
        )
        
        
class FastRcnnPredictHead(nn.Sequential):
    """
    Code from mmdetection + torchvision
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes: int = 80, reg_class_agnostic: bool = False):
        super().__init__()
        self.box_layer = nn.Linear(
            in_channels, 4 if reg_class_agnostic else num_classes * 4)
        self.score_layer = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]

        x = x.flatten(start_dim=1)
        bbox_deltas = self.box_layer(x)
        scores = self.score_layer(x)

        return scores, bbox_deltas
        
        
class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d[f"mask_fcn{layer_idx}"] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation
            )
            d[f"relu{layer_idx}"] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super().__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class CascadeMaskRCNNPretrained(Model):
    config_class = CascadeMaskRCNNConfig
    base_model_prefix = 'cascade_mask_rcnn'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = CascadeMaskRCNNModel(config)

        pretrained_file = super().get_pretrained_from_file(
            name_or_path, cache_dir=cls.config_class.cache_dir)
        model.load_weights(pretrained_file)

        return model
    
    
class CascadeMaskRCNNModel(CascadeMaskRCNNPretrained):
    model_name = 'cascade_mask_rcnn'
    
    def __init__(
        self, 
        config: CascadeMaskRCNNConfig,
        **kwargs
    ):
        super().__init__(config)
        self.config = config
        self.min_size = config.min_size
        self.max_size = config.max_size
        self.num_classes = config.num_classes
        self.preserve_aspect_ratio = config.preserve_aspect_ratio
        
        self.update_config(config)
        
        self.backbone = self.get_resnet_backbone(config.backbone_name)
        self.neck = CascadeMaskRCNNNeck(
            config=config,
            channels=self.backbone.channels,
        )
        self.fpn_last_downsample = nn.MaxPool2d(2)
        
        anchor_sizes = tuple((anchor,) for anchor in config.anchor_sizes)
        aspect_ratios = (config.aspect_ratios,) * len(config.anchor_sizes)
        
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        
        out_channels = self.neck.channels[-1]
        rpn_head = RpnHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0],
        )
        
        rpn_pre_nms_top_n = {
            'training': config.rpn_pre_nms_top_n_train, 
            'testing': config.rpn_pre_nms_top_n_test,
        }
        rpn_post_nms_top_n = {
            'training': config.rpn_post_nms_top_n_train, 
            'testing': config.rpn_post_nms_top_n_test,
        }
                
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            config.rpn_box_coder_weights,
            config.rpn_fg_iou_thresh,
            config.rpn_bg_iou_thresh,
            config.rpn_batch_size_per_image,
            config.rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            config.rpn_nms_thresh,
            score_thresh=config.rpn_score_thresh,
        )
        
        box_roi_pool_1 = MultiScaleRoIAlign(
            featmap_names=config.box_roi_pool_feat_names,
            output_size=config.box_roi_pool_out_size,
            sampling_ratio=config.box_roi_pool_sample_ratio,
        )

        box_head_1 = LinearHead(
            out_channels * config.box_roi_pool_out_size ** 2,
            config.representation_size,
        )

        box_predictor_1 = FastRcnnPredictHead(
            config.representation_size,
            self.num_classes,
            config.box_roi_reg_class_agnostic,
        )
        
        mask_roi_pool_1 = MultiScaleRoIAlign(
            featmap_names=config.mask_roi_pool_feat_names,
            output_size=config.mask_roi_pool_out_size,
            sampling_ratio=config.mask_roi_pool_sample_ratio,
        )
        
        mask_head_1 = MaskRCNNHeads(
            config.mask_head_in_channels,
            config.mask_layers,
            config.mask_dilation,
        )
        
        mask_predictor_1 = MaskRCNNPredictor(
            config.mask_predictor_in_channels, 
            config.mask_dim_reduced, 
            self.num_classes-1,
        )

        self.roi_heads_1 = RoiHeads(
            box_roi_pool_1,
            box_head_1,
            box_predictor_1,
            config.box_fg_iou_thresh_1,
            config.box_bg_iou_thresh_1,
            config.box_batch_size_per_image,
            config.box_positive_fraction,
            config.bbox_reg_weights_1,
            config.box_score_thresh,
            config.box_nms_thresh_1,
            config.box_detections_per_img,
            mask_roi_pool_1,
            mask_head_1,
            mask_predictor_1,
        )
        
        box_roi_pool_2 = MultiScaleRoIAlign(
            featmap_names=config.box_roi_pool_feat_names,
            output_size=config.box_roi_pool_out_size,
            sampling_ratio=config.box_roi_pool_sample_ratio,
        )

        box_head_2 = LinearHead(
            out_channels * config.box_roi_pool_out_size ** 2,
            config.representation_size,
        )

        box_predictor_2 = FastRcnnPredictHead(
            config.representation_size,
            self.num_classes,
            config.box_roi_reg_class_agnostic,
        )
        
        mask_roi_pool_2 = MultiScaleRoIAlign(
            featmap_names=config.mask_roi_pool_feat_names,
            output_size=config.mask_roi_pool_out_size,
            sampling_ratio=config.mask_roi_pool_sample_ratio,
        )
        
        mask_head_2 = MaskRCNNHeads(
            config.mask_head_in_channels,
            config.mask_layers,
            config.mask_dilation,
        )
        
        mask_predictor_2 = MaskRCNNPredictor(
            config.mask_predictor_in_channels, 
            config.mask_dim_reduced, 
            self.num_classes-1,
        )

        self.roi_heads_2 = RoiHeads(
            box_roi_pool_2,
            box_head_2,
            box_predictor_2,
            config.box_fg_iou_thresh_2,
            config.box_bg_iou_thresh_2,
            config.box_batch_size_per_image,
            config.box_positive_fraction,
            config.bbox_reg_weights_2,
            config.box_score_thresh,
            config.box_nms_thresh_2,
            config.box_detections_per_img,
            mask_roi_pool_2,
            mask_head_2,
            mask_predictor_2,
        )
        
        box_roi_pool_3 = MultiScaleRoIAlign(
            featmap_names=config.box_roi_pool_feat_names,
            output_size=config.box_roi_pool_out_size,
            sampling_ratio=config.box_roi_pool_sample_ratio,
        )

        box_head_3 = LinearHead(
            out_channels * config.box_roi_pool_out_size ** 2,
            config.representation_size,
        )

        box_predictor_3 = FastRcnnPredictHead(
            config.representation_size,
            self.num_classes,
            config.box_roi_reg_class_agnostic,
        )
        
        mask_roi_pool_3 = MultiScaleRoIAlign(
            featmap_names=config.mask_roi_pool_feat_names,
            output_size=config.mask_roi_pool_out_size,
            sampling_ratio=config.mask_roi_pool_sample_ratio,
        )
        
        mask_head_3 = MaskRCNNHeads(
            config.mask_head_in_channels,
            config.mask_layers,
            config.mask_dilation,
        )
        
        mask_predictor_3 = MaskRCNNPredictor(
            config.mask_predictor_in_channels, 
            config.mask_dim_reduced, 
            self.num_classes-1,
        )

        self.roi_heads_3 = RoiHeads(
            box_roi_pool_3,
            box_head_3,
            box_predictor_3,
            config.box_fg_iou_thresh_3,
            config.box_bg_iou_thresh_3,
            config.box_batch_size_per_image,
            config.box_positive_fraction,
            config.bbox_reg_weights_3,
            config.box_score_thresh,
            config.box_nms_thresh_3,
            config.box_detections_per_img,
            mask_roi_pool_3,
            mask_head_3,
            mask_predictor_3,
        )
    
    def get_resnet_backbone(self, backbone_name):
        backbones = {
            'resnet18': resnet18(),
            'resnet34': resnet34(),
            'resnet50': resnet50(),
            'resnet101': resnet101(),
        }
        return backbones[backbone_name]
        
    def init_weights(self, path):
        self.backbone.from_pretrained(path)

        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, images):
        images = self.check_inputs(images)
        
        images, image_sizes, resized_sizes = self.resize_inputs(
            images, self.max_size, preserve_aspect_ratio=self.preserve_aspect_ratio)
        
        outputs = self.backbone(images)
        outputs = self.neck(outputs)
        outputs.append(self.fpn_last_downsample(outputs[-1]))
        outputs = {str(i): o for i, o in enumerate(outputs)}
        
        if self.training:
            raise NotImplementedError
        else:
            proposals = self.rpn(images, resized_sizes, outputs)
            detections_1 = self.roi_heads_1(outputs, proposals, resized_sizes)
            b1 = [d['boxes'] for d in detections_1]
            
            detections_2 = self.roi_heads_2(outputs, b1, resized_sizes)
            b2 = [d['boxes'] for d in detections_2]
            
            detections_3 = self.roi_heads_3(outputs, b2, resized_sizes)
            
        return detections_1, detections_2, detections_3

    def load_weights(self, path):
        # state_dict = torch.load(path)
        state_dict = load_state_dict_from_url(path)['state_dict']
        
        for key in list(state_dict.keys()):
            p = key.split('.')
            if p[0] == 'backbone':
                if p[1] == 'conv1':
                    new_key = f'backbone.conv.{p[2]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'bn1':
                    new_key = f'backbone.bn.{p[2]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[3].startswith('conv'):
                    new_key = f'backbone.layers.{int(p[1][-1])-1}.{p[2]}.{p[3]}.0.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[3] == 'downsample':
                    if p[4] == '0':
                        new_key = f'backbone.layers.{int(p[1][-1])-1}.{p[2]}.{p[3]}.0.0.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                    elif p[4] == '1':
                        new_key = f'backbone.layers.{int(p[1][-1])-1}.{p[2]}.{p[3]}.1.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                else:
                    new_key = f'backbone.layers.{int(p[1][-1])-1}.{p[2]}.{p[3]}.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
            elif p[0] == 'neck':
                new_p2 = {'0': '3', '1': '2', '2': '1', '3': '0'}
                if p[1].startswith('lateral'):
                    new_key = f'neck.lateral_layers.{new_p2[p[2]]}.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                # TODO: Should I use new_p2 in below condition?
                elif p[1].startswith('fpn'):
                    # new_key = f'neck.predict_layers.{new_p2[p[2]]}.{p[-1]}'
                    new_key = f'neck.predict_layers.{p[2]}.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
            elif p[0].startswith('rpn'):
                if p[1].endswith('conv'):
                    new_key = f'rpn.head.conv.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1].endswith('cls'):
                    new_key = f'rpn.head.score_layer.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1].endswith('reg'):
                    new_key = f'rpn.head.box_layer.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
            elif p[0] == 'roi_head':
                if p[3] == 'fc_cls':
                    new_key = f'roi_heads_{int(p[2])+1}.box_predictor.score_layer.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[3] == 'fc_reg':
                    new_key = f'roi_heads_{int(p[2])+1}.box_predictor.box_layer.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[3] == 'shared_fcs':
                    if p[4] == '0':
                        new_key = f'roi_heads_{int(p[2])+1}.box_head.layers.1.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                    elif p[4] == '1':
                        new_key = f'roi_heads_{int(p[2])+1}.box_head.layers.3.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                elif p[3] == 'convs':
                    new_key = f'roi_heads_{int(p[2])+1}.mask_head.mask_fcn{int(p[4])+1}.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[3] == 'upsample':
                    new_key = f'roi_heads_{int(p[2])+1}.mask_predictor.conv5_mask.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[3] == 'conv_logits':
                    new_key = f'roi_heads_{int(p[2])+1}.mask_predictor.mask_fcn_logits.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                    
        self.load_state_dict(state_dict)
