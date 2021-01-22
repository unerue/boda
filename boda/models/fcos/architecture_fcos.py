from boda.models.backbone_resnet import resnet101
import math
from typing import List

import torch
from torch import nn, Tensor
from ..neck_fpn import FeaturePyramidNetwork


class FcosPredictNeck(FeaturePyramidNetwork):
    ...


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FcosPredictHead(nn.Module):
    def __init__(self, config, channels, seletec_layers, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        self.config = config
        # TODO: Implement the sigmoid version first.
        self.num_classes = 80
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.num_box_layers = 4
        self.num_score_layers = 4
        self.num_share_layers = 0

        # head_configs = {"cls": (4, False),
        #                 "bbox": (4, False),
        #                 "share": (0, False)}
        # cfg.MODEL.FCOS.NORM = 'GN'
        # norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.box_layers = nn.ModuleList()
        for _ in self.num_box_layers:
            self.box_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU()
                )
            )

        self.score_layers = nn.ModuleList()
        for _ in self.num_score_layers:
            self.score_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU()
                )
            )

        self.share_layers = nn.ModuleList()
        for _ in self.num_share_layers:
            self.share_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU()
                )
            )

        self.pred_box_layer = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.pred_score_layer = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )

        self.ctrness_layer = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])

    def forward(self, inputs: List[Tensor]):
        boxes = []
        scores = []
        ctrness = []
        for i, feature in enumerate(inputs):
            feature = self.share_layers(feature)
            pred_boxes = self.box_layers(feature)
            pred_scores = self.score_layers(feature)

            scores.append(self.pred_score_layer(pred_scores))
            ctrness.append(self.pred_ctrness_layer(pred_boxes))
            if self.scales it not None:
                pred_boxes = self.scales[i](pred_boxes)
            boxes.append(self.pred_box_layer(pred_boxes))

        return boxes, scores, ctrness


class FcosModel(nn.Module):
    def __init__(
        self,
        config,
        input_shape: Dict[str, ShapeSpec]
    ) -> None:
        super().__init__()
        # fmt: off
        self.in_features = ["p3", "p4", "p5", "p6", "p7"]  # selected_layers 
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.center_sample = True
        self.strides = [8, 16, 32, 64, 128]
        self.radius = 1.5
        self.pre_nms_thresh_train = 0.05
        self.pre_nms_thresh_test = 0.05
        self.pre_nms_topk_train = 1000
        self.pre_nms_topk_test = 1000
        self.nms_thresh = 0.6
        self.post_nms_topk_train = 100
        self.post_nms_topk_test = 50
        self.thresh_with_ctr = False
        self.mask_on = True
        # fmt: on
        self.iou_loss = IOULoss('giou')
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in [64, 128, 256, 512]:
            soi.append([prev_size, s])
            prev_size = s

        soi.append([prev_size, 100000000])
        self.sizes_of_interest = soi
        self.backbone = resnet101()
        self.neck = FcosPredictNeck(config, self.backbone.channels, selected_layers=[1, 2, 3])
        self.heads = FcosPredictHead(config, self.neck.channels, self.neck.selected_layers)

    def forward(self, inputs):
        pass

    def forward(self, images, features, gt_instances):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, bbox_towers = self.fcos_head(features)

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        outputs = FCOSOutputs(
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.fcos_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            self.thresh_with_ctr,
            gt_instances,
        )

        if self.training:
            losses, _ = outputs.losses()
            if self.mask_on:
                proposals = outputs.predict_proposals()
                return proposals, losses
            else:
                return None, losses
        else:
            proposals = outputs.predict_proposals()
            return proposals, {}


