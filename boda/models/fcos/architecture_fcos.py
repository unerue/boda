import os
from typing import List, Union, Sequence

import torch
from torch import nn, Tensor
from ...base_architecture import Neck, Head, Model
from ..neck_fpn import FeaturePyramidNetwork
from .configuration_fcos import FcosConfig
from ..backbone_resnet import resnet101


class FcosPredictNeck(FeaturePyramidNetwork):
    def __init__(
        self,
        config: FcosConfig = None,
        channels: Sequence[int] = [256, 512, 1024, 2048],
        selected_layers: Sequence[int] = [1, 2, 3],
        fpn_channels: int = 256,
        num_extra_fpn_layers: int = 1
    ) -> None:
        super().__init__(
            config,
            channels,
            selected_layers,
            fpn_channels,
            num_extra_fpn_layers
        )


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FcosPredictHead(nn.Module):
    def __init__(
        self,
        config,
        channels,
        seletec_layers,
        fpn_channels: int = 256
    ) -> None:
        """
        Args:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        self.config = config
        self.channels = channels,
        self.selected_layers = seletec_layers
        self.num_classes = 80
        self.fpn_channels = fpn_channels
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.num_box_layers = 4
        self.num_score_layers = 4
        self.num_share_layers = 1

        box_layers = []
        for _ in range(self.num_box_layers):
            box_layers += [
                nn.Conv2d(
                    self.fpn_channels, self.fpn_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True),
                nn.GroupNorm(32, self.fpn_channels),
                nn.ReLU()
            ]
        self.add_module('box_layers', nn.Sequential(*box_layers))

        score_layers = []
        for _ in range(self.num_score_layers):
            score_layers += [
                nn.Conv2d(
                    self.fpn_channels, self.fpn_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True),
                nn.GroupNorm(32, self.fpn_channels),
                nn.ReLU()
            ]
        self.add_module('score_layers', nn.Sequential(*score_layers))

        share_layers = []
        for _ in range(self.num_share_layers):
            share_layers += [
                nn.Conv2d(
                    self.fpn_channels, self.fpn_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True),
                nn.GroupNorm(32, self.fpn_channels),
                nn.ReLU()
            ]
        self.add_module('share_layers', nn.Sequential(*share_layers))

        self.pred_box_layer = nn.Conv2d(
            self.fpn_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.pred_score_layer = nn.Conv2d(
            self.fpn_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )

        self.pred_center_layer = nn.Conv2d(
            self.fpn_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])

    def forward(self, inputs: List[Tensor]):
        boxes = []
        scores = []
        centerness = []
        print(len(inputs))
        print(len(self.scales))
        for i, feature in enumerate(inputs):
            feature = self.share_layers(feature)
            pred_boxes = self.box_layers(feature)
            pred_scores = self.score_layers(feature)

            scores.append(self.pred_score_layer(pred_scores))
            centerness.append(self.pred_center_layer(pred_boxes))
            if self.scales is not None:
                pred_boxes = self.scales[i](pred_boxes)

            boxes.append(self.pred_box_layer(pred_boxes))

        return boxes, scores, centerness


class FcosPretrained(Model):
    config_class = FcosConfig
    base_model_prefix = 'fcos'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = FcosModel(config)

        return model


class FcosModel(FcosPretrained):
    def __init__(
        self,
        config,
        # channels,
        selected_layers=[1, 2, 3],
        strides=[8, 16, 32, 64, 128],
    ) -> None:
        super().__init__(config)
        self.config = config
        # self.channels = channels
        self.selected_layers = selected_layers
        self.fpn_strides = [8, 16, 32, 64, 128]

        self.update_config(config)

        self.backbone = resnet101()
        self.neck = FcosPredictNeck(
            config,
            [self.backbone.channels[i] for i in self.selected_layers],
            selected_layers=[1, 2, 3],
            num_extra_fpn_layers=2
        )

        self.heads = FcosPredictHead(
            config,
            self.neck.channels,
            self.neck.selected_layers
        )

    def forward(self, inputs):
        inputs = self.check_inputs(inputs)
        outputs = self.backbone(inputs)
        outputs = [outputs[i] for i in self.selected_layers]
        outputs = self.neck(outputs)
        o1, o2, o3 = self.heads(outputs)
        return o1, o2, o3

    # def forward(self, images, features, gt_instances):
    #     """
    #     Arguments:
    #         images (list[Tensor] or ImageList): images to be processed
    #         targets (list[BoxList]): ground-truth boxes present in the image (optional)

    #     Returns:
    #         result (list[BoxList] or dict[Tensor]): the output from the model.
    #             During training, it returns a dict[Tensor] which contains the losses.
    #             During testing, it returns list[BoxList] contains additional fields
    #             like `scores`, `labels` and `mask` (for Mask R-CNN models).

    #     """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, bbox_towers = self.fcos_head(features)

        # if self.training:
        #     pre_nms_thresh = self.pre_nms_thresh_train
        #     pre_nms_topk = self.pre_nms_topk_train
        #     post_nms_topk = self.post_nms_topk_train
        # else:
        #     pre_nms_thresh = self.pre_nms_thresh_test
        #     pre_nms_topk = self.pre_nms_topk_test
        #     post_nms_topk = self.post_nms_topk_test

        # outputs = FCOSOutputs(
        #     images,
        #     locations,
        #     logits_pred,
        #     reg_pred,
        #     ctrness_pred,
        #     self.focal_loss_alpha,
        #     self.focal_loss_gamma,
        #     self.iou_loss,
        #     self.center_sample,
        #     self.sizes_of_interest,
        #     self.strides,
        #     self.radius,
        #     self.fcos_head.num_classes,
        #     pre_nms_thresh,
        #     pre_nms_topk,
        #     self.nms_thresh,
        #     post_nms_topk,
        #     self.thresh_with_ctr,
        #     gt_instances,
        # )

        # if self.training:
        #     losses, _ = outputs.losses()
        #     if self.mask_on:
        #         proposals = outputs.predict_proposals()
        #         return proposals, losses
        #     else:
        #         return None, losses
        # else:
        #     proposals = outputs.predict_proposals()
        #     return proposals, {}
