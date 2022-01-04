import os
from typing import Union

import torch
from torch import nn

from ...base_architecture import Model
from .configuration_efficientdet import EfficientDetConfig
from ..backbone_efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2
from ..backbone_efficientnet import efficientnet_b3, efficientnet_b4, efficientnet_b5
from ..backbone_efficientnet import efficientnet_b6, efficientnet_b7
from ..bifpn import BiFPN
from ...ops.misc import SeparableConvBlock, MemoryEfficientSwish
from .anchor import Anchors
from .utils import Resizer, BBoxTransform, ClipBoxes, invert_affine, postprocess


class EfficientDetPredictNeck(BiFPN):
    def __init__(
        self,
        num_channels,
        conv_channels,
        first_time=False,
        epsilon=0.0001,
        attention=True,
        use_p8=False
    ):
        super().__init__(
            num_channels,
            conv_channels,
            first_time=first_time,
            epsilon=epsilon,
            attention=attention,
            use_p8=use_p8
        )
        
        
class EfficientDetBoxPredictHead(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5):
        super().__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [
                SeparableConvBlock(in_channels, in_channels, norm=False, activation=False)
                for i in range(num_layers)
            ]
        )
        self.bn_list = nn.ModuleList(
            [
                nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) 
                for j in range(pyramid_levels)
            ]
        )
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats
    
    
class EfficientDetClassPredictHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [
                SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) 
                for i in range(num_layers)
            ]
        )
        self.bn_list = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
                        for i in range(num_layers)
                    ]
                ) for j in range(pyramid_levels)
            ]
        )
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats


class EfficientDetPretrained(Model):
    config_class = EfficientDetConfig
    base_model_prefix = 'efficientdet'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = EfficientDetModel(config)
        
        pretrained_file = super().get_pretrained_from_file(
            name_or_path, cache_dir=cls.config_class.cache_dir)
        model.load_weights(pretrained_file)
        
        return model
    

class EfficientDetModel(EfficientDetPretrained):
    model_name = 'efficientdet'
    
    def __init__(
        self,
        config: EfficientDetConfig,
        **kwargs
    ) -> None:
        super().__init__(config)
        self.config = config
        self.num_classes = kwargs.get('num_classes', config.num_classes)
        self.input_size = kwargs.get('input_size', config.input_size)
        self.fpn_channels = config.fpn_channels
        self.aspect_ratios = config.aspect_ratios
        self.scales = config.scales
        
        self.update_config(config)
        
        self.backbone = self.get_efficientnet_backbone(config.backbone_name)
        
        self.neck = nn.Sequential(
            *[
                EfficientDetPredictNeck(
                    self.fpn_channels,
                    config.conv_channel_coef,
                    True if _ == 0 else False,
                    attention=True if self.compound_coef < 6 else False,
                    use_p8=self.compound_coef > 7
                ) for _ in range(config.fpn_cell_repeat)
            ]
        )
        
        self.box_net = EfficientDetBoxPredictHead(
            in_channels=self.fpn_channels,
            num_anchors=config.num_anchors,
            num_layers=config.box_class_repeat,
            pyramid_levels=config.pyramid_level,
        )
        
        self.class_net = EfficientDetClassPredictHead(
            in_channels=self.fpn_channels,
            num_anchors=config.num_anchors,
            num_classes=self.num_classes,
            num_layers=config.box_class_repeat,
            pyramid_levels=config.pyramid_level,
        )
        
        self.anchors = Anchors(
            anchor_scale=config.anchor_scale,
            pyramid_levels=(
                torch.arange(config.pyramid_level) + 3
            ).tolist(),
            **kwargs
        )
        
        self.resizer = Resizer(self.input_size)
        
    def get_efficientnet_backbone(self, backbone_name):
        backbones = {
            'efficientnet_b0': efficientnet_b0(),
            'efficientnet_b1': efficientnet_b1(),
            'efficientnet_b2': efficientnet_b2(),
            'efficientnet_b3': efficientnet_b3(),
            'efficientnet_b4': efficientnet_b4(),
            'efficientnet_b5': efficientnet_b5(),
            'efficientnet_b6': efficientnet_b6(),
            'efficientnet_b7': efficientnet_b7(),
        }
        return backbones[backbone_name]
    
    def init_weights(self, path):
        self.backbone.from_pretrained(path)

        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()
                    
    def forward(self, inputs):
        '''
        Args:
            inputs: List[Tensor] -> Tensor
        '''
        outputs = []
        old_whs = []
        for image in inputs:
            image = self.resizer(image)
            outputs.append(image)
            old_whs.append([image.shape[-1], image.shape[-2]])
        inputs = torch.stack(outputs)  # Tensor
        
        _, p3, p4, p5 = self.backbone(inputs)

        features = (p3, p4, p5)
        features = self.neck(features)

        regression = self.box_net(features)
        classification = self.class_net(features)
        anchors = self.anchors(inputs, inputs.dtype)
        
        if self.training:
            return features, regression, classification, anchors
        else:
            preds = postprocess(
                inputs,
                anchors,
                regression, 
                classification,
                regressBoxes=BBoxTransform(), 
                clipBoxes=ClipBoxes(),
                threshold=0.05, 
                iou_threshold=0.5
            )
            
            preds = invert_affine(old_whs, self.input_size, preds)
            
            return preds

    def load_weights(self, path):
        new_backbone_idx = {
            '0.0': '0', '1.0': '1', '1.1': '2',
            '2.0': '3', '2.1': '4', '3.0': '5', 
            '3.1': '6', '3.2': '7', '4.0': '8', 
            '4.1': '9', '4.2': '10', '5.0': '11', 
            '5.1': '12', '5.2': '13', '5.3': '14',
            '6.0': '15',
        }
        
        new_fpn_idx = {
            '0.combine.edge_weights': 'p6_w1',
            '1.combine.edge_weights': 'p5_w1',
            '2.combine.edge_weights': 'p4_w1',
            '3.combine.edge_weights': 'p3_w1',
            '4.combine.edge_weights': 'p4_w2',
            '5.combine.edge_weights': 'p5_w2',
            '6.combine.edge_weights': 'p6_w2',
            '7.combine.edge_weights': 'p7_w2',
            '1.combine.resample': 'p5_down_channel',
            '2.combine.resample': 'p4_down_channel',
            '3.combine.resample': 'p3_down_channel',
            '4.combine.resample': 'p4_down_channel_2',
            '5.combine.resample': 'p5_down_channel_2',
            '0.after_combine.conv': 'conv6_up',
            '1.after_combine.conv': 'conv5_up',
            '2.after_combine.conv': 'conv4_up',
            '3.after_combine.conv': 'conv3_up',
            '4.after_combine.conv': 'conv4_down',
            '5.after_combine.conv': 'conv5_down',
            '6.after_combine.conv': 'conv6_down',
            '7.after_combine.conv': 'conv7_down',
        }
        
        new_classnet_bn_idx = {
            '0.0': '0.0', '0.1': '0.1', '0.2': '0.2',
            '0.3': '1.0', '0.4': '1.1', '1.0': '1.2',
            '1.1': '2.0', '1.2': '2.1', '1.3': '2.2',
            '1.4': '3.0', '2.0': '3.1', '2.1': '3.2',
            '2.2': '4.0', '2.3': '4.1', '2.4': '4.2',
        }
        
        state_dict = torch.load(path)
        
        for key in list(state_dict.keys()):
            p = key.split('.')
            if p[0] == 'backbone':
                if p[1] == 'blocks':
                    default = f'{p[0]}.layers'
                    old_backbone_idx = f'{p[2]}.{p[3]}'
                    if old_backbone_idx == '0.0':
                        if p[4] == 'conv_dw':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.0.0.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'bn1':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.0.1.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'se':
                            if p[5] == 'conv_reduce':
                                new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.1.fc1.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                            elif p[5] == 'conv_expand':
                                new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.1.fc2.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'conv_pw':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.2.0.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'bn2':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.2.1.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                    else:
                        if p[4] == 'conv_pw':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.0.0.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'bn1':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.0.1.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'conv_dw':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.1.0.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'bn2':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.1.1.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'se':
                            if p[5] == 'conv_reduce':
                                new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.2.fc1.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                            elif p[5] == 'conv_expand':
                                new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.2.fc2.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'conv_pwl':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.3.0.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[4] == 'bn3':
                            new_key = f'{default}.{new_backbone_idx[old_backbone_idx]}.block.3.1.{p[-1]}'
                            state_dict[new_key] = state_dict.pop(key)
                else:
                    if p[1] == 'conv_stem':
                        new_key = f'{p[0]}.firstconv_layer.0.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                    elif p[1] == 'bn1':
                        new_key = f'{p[0]}.firstconv_layer.1.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
            elif p[0] == 'fpn':
                if p[1] == 'resample':
                    if p[4] == 'conv':
                        new_key = f'neck.0.p5_to_p6.0.{p[4]}.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                    elif p[4] == 'bn':
                        new_key = f'neck.0.p5_to_p6.1.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'cell':
                    old_fpn_idx = f'{p[4]}.{p[5]}.{p[6]}'
                    if p[2] == '0':
                        if p[5] == 'combine':
                            if p[6] == 'edge_weights':
                                new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}'
                                state_dict[new_key] = state_dict.pop(key)
                            elif p[6] == 'resample':
                                if p[9] == 'conv':
                                    new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}.0.{p[9]}.{p[-1]}'
                                    state_dict[new_key] = state_dict.pop(key)
                                elif p[9] == 'bn':
                                    new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}.1.{p[-1]}'
                                    state_dict[new_key] = state_dict.pop(key)
                        elif p[5] == 'after_combine':
                            if p[7] == 'conv_dw':
                                new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}.depthwise_conv.conv.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                            elif p[7] == 'conv_pw':
                                new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}.pointwise_conv.conv.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                            elif p[7] == 'bn':
                                new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}.bn.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                    else:
                        if p[5] == 'combine':
                            new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}'
                            state_dict[new_key] = state_dict.pop(key)
                        elif p[5] == 'after_combine':
                            if p[7] == 'conv_dw':
                                new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}.depthwise_conv.conv.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                            elif p[7] == 'conv_pw':
                                new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}.pointwise_conv.conv.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
                            elif p[7] == 'bn':
                                new_key = f'neck.{p[2]}.{new_fpn_idx[old_fpn_idx]}.bn.{p[-1]}'
                                state_dict[new_key] = state_dict.pop(key)
            else:
                if p[1] == 'conv_rep':
                    if p[3] == 'conv_dw':
                        new_key = f'{p[0]}.conv_list.{p[2]}.depthwise_conv.conv.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                    elif p[3] == 'conv_pw':
                        new_key = f'{p[0]}.conv_list.{p[2]}.pointwise_conv.conv.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'bn_rep':
                    old_classnet_bn_idx = f'{p[2]}.{p[3]}'
                    new_key = f'{p[0]}.bn_list.{new_classnet_bn_idx[old_classnet_bn_idx]}.{p[-1]}'
                    state_dict[new_key] = state_dict.pop(key)
                elif p[1] == 'predict':
                    if p[2] == 'conv_dw':
                        new_key = f'{p[0]}.header.depthwise_conv.conv.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                    elif p[2] == 'conv_pw':
                        new_key = f'{p[0]}.header.pointwise_conv.conv.{p[-1]}'
                        state_dict[new_key] = state_dict.pop(key)
                        
        self.load_state_dict(state_dict)
