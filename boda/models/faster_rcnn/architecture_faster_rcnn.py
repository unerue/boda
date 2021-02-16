# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
from typing import Union
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torchvision.ops import MultiScaleRoIAlign

from ...base_architecture import Neck, Head, Model
from .configuration_faster_rcnn import FasterRcnnConfig
from .anchor_generator import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork


class FasterRcnnNeck(Neck):
    def __init__(self):
        ...


class FasterRcnnHead(Head):
    def __init__(self):
        ...


class FasterRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for i, layer_features in enumerate(layers, 1):
            d[f'mask{i}'] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d[f'relu{i}'] = nn.ReLU()
            next_feature = layer_features

        super().__init__(d)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class FasterRcnnPredictor(nn.Sequential):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

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
    """Mask R-CNN
    """
    model_name = 'faster_rcnn'

    def __init__(
        self,
        config,
        backbone,
        neck,
        head,
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
        num_classes = 91
        self.backbone = backbone
        self.neck = neck

        # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        anchor_sizes = [[anchor] for anchor in anchor_sizes]
        # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        aspect_ratios = [aspect_ratios] * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        rpn_head = RPNHead(
                backbone.channels, rpn_anchor_generator.num_anchors_per_location()[0]
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
                sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024

        self.box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size
        )

        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            self.num_classes
        )

        self.roi_heads = RoIHeads(
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

    def forward(self, inputs):
        inputs = self.check_inputs(inputs)

        outputs = self.backbone(inputs)
        outputs = {str(i): o for i, o in outputs}

        # if isinstance(features, torch.Tensor):
        #     features = OrderedDict([('0', features)])
        if self.training:
            # proposals, proposal_losses = self.rpn(images, features, targets)
            # detections, detector_losses = \
            #     self.roi_heads(features, proposals, images.image_sizes, targets)
            pass
        else:
            proposals = self.rpn(inputs, outputs)
            detections = self.roi_heads(outputs, proposals, inputs.size)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes)

        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)

        # if torch.jit.is_scripting():
        #     if not self._has_warned:
        #         warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
        #         self._has_warned = True
        #     return losses, detections
        # else:
        #     return self.eager_outputs(losses, detections)
        return outputs
