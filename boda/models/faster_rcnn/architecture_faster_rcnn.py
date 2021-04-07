# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
import os
import sys
from collections import OrderedDict
from typing import List, Union, Sequence

import torch
from torch import nn, Tensor
from torchvision.ops import MultiScaleRoIAlign

from ...base_architecture import Neck, Head, Model
from .configuration_faster_rcnn import FasterRcnnConfig
from .anchor_generator import AnchorGenerator
from .rpn import RpnHead, RegionProposalNetwork
from .roi_heads import RoiHeads
from ..neck_fpn import FeaturePyramidNetwork
from ..backbone_resnet import resnet50
from ._transform import RcnnTransform


class FasterRcnnNeck(FeaturePyramidNetwork):
    def __init__(
        self,
        config,
        channels: Sequence[int],
        selected_layers: Sequence[int] = [1, 2, 3],
        fpn_channels: int = 256,
        extra_layers: bool = True,
        num_extra_fpn_layers: int = 2,
    ) -> None:
        super().__init__(
            config,
            channels,
            selected_layers,
            fpn_channels,
            extra_layers,
            num_extra_fpn_layers
        )


class LinearHead(nn.Sequential):
    def __init__(self, in_channels, representation_size):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class FastRcnnPredictHead(nn.Sequential):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes: int = 80):
        super().__init__()
        self.box_layer = nn.Linear(in_channels, num_classes * 4)
        self.score_layer = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]

        x = x.flatten(start_dim=1)
        bbox_deltas = self.box_layer(x)
        scores = self.score_layer(x)

        return scores, bbox_deltas


class FasterRcnnPretrained(Model):
    config_class = FasterRcnnConfig
    base_model_prefix = 'mask_rcnn'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = FasterRcnnModel(config)
        # model.state_dict(torch.load('test.pth'))
        return model


class FasterRcnnModel(FasterRcnnPretrained):
    """Faster R-CNN
    """
    model_name = 'faster_rcnn'

    def __init__(
        self,
        config,
        # backbone,
        # neck,
        # head,
        min_size=800,
        max_size=1333,
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        anchor_sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0)
    ) -> None:
        super().__init__(config)
        self.num_classes = 91

        self.transform = RcnnTransform(800, 1333, None, None)
        self.backbone = resnet50()
        self.neck = FasterRcnnNeck(config, self.backbone.channels)

        anchor_sizes = [[anchor] for anchor in anchor_sizes]
        aspect_ratios = [aspect_ratios] * len(anchor_sizes)

        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        out_channels = self.neck.channels[-1]
        rpn_head = RpnHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n = {
            'training': rpn_pre_nms_top_n_train, 'testing': rpn_pre_nms_top_n_test
        }
        rpn_post_nms_top_n = {
            'training': rpn_post_nms_top_n_train, 'testing': rpn_post_nms_top_n_test
        }

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh
        )

        box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024

        box_head = LinearHead(
            out_channels * resolution ** 2,
            representation_size
        )

        representation_size = 1024
        box_predictor = FastRcnnPredictHead(
            representation_size,
            self.num_classes
        )

        self.roi_heads = RoiHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img
        )

    def forward(self, images: List[Tensor]):
        """
        Args:
            inputs (List[Tensor]): Original image size; do not resize image
            sizes (List[int, int])
        """
        # images = self.check_inputs(images)
        # images, targets = self.transform(images)
        images, image_sizes, targets = self.transform(images)
        # [(1920, 1080), ()]
        # outputs = self.backbone(images.tensors)
        outputs = self.backbone(images)  # resnet50, 101, 154, 18, 32
        # [2, 3, 4, 5]
        outputs = self.neck(outputs)
        # [1, 2, 3, 4, 5]
        outputs = {str(i): o for i, o in enumerate(outputs)}

        if self.training:
            raise NotImplementedError
        else:
            # proposals = self.rpn(images.tensors, images.image_sizes, outputs)
            # detections = self.roi_heads(outputs, proposals, images.image_sizes)
            proposals = self.rpn(images, image_sizes, outputs)
            detections = self.roi_heads(outputs, proposals, image_sizes)
            # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections
