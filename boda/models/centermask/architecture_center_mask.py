from typing import Union, Sequence

import torch
from torch import nn, Tensor
from ...base_architecture import Neck, Head, Model
from ..neck_fpn import FeaturePyramidNetwork
from .configuration_center_mask import CenterMaskConfig


class CenterMaskPredictNeck(FeaturePyramidNetwork):
    def __init__(
        self,
        config: CenterMaskConfig,
        channels: Sequence[int],
        selected_layers: Sequence[int] = [1, 2, 3],
        fpn_channels: int = 256,
        num_extra_fpn_layers: int = 2,
    ) -> None:
        super().__init__(
            config,
            channels,
            selected_layers,
            fpn_channels,
            num_extra_fpn_layers
        )


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)


class CenterMaskRoiHeads(nn.Module):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches  masks directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_mask_head(cfg)
        self._init_mask_iou_head(cfg)
        self._init_keypoint_head(cfg, input_shape)


    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON # True
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        assign_crit       = cfg.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION # ratio

        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = RoiPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            assign_crit=assign_crit,
        )
        self.mask_head = SpatialAttentionMaskHead(
            cfg,
            ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution))


    def _init_mask_iou_head(self, cfg):
        # fmt: off
        self.maskiou_on     = cfg.MODEL.MASKIOU_ON # True
        if not self.maskiou_on:
            return
        in_channels         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        pooler_resolution   = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.maskiou_weight = cfg.MODEL.MASKIOU_LOSS_WEIGHT

        # fmt : on

        self.maskiou_head = build_maskiou_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )


    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        self.kp_in_features = cfg.MODEL.ROI_KEYPOINT_HEAD.IN_FEATURES
        pooler_resolution   = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales       = tuple(1.0 / input_shape[k].stride for k in self.kp_in_features)  # noqa
        sampling_ratio      = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type         = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        assign_crit         = cfg.MODEL.ROI_KEYPOINT_HEAD.ASSIGN_CRITERION
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.kp_in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            assign_crit=assign_crit,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            if self.maskiou_on:
                losses, mask_features, selected_mask, labels, maskiou_targets = self._forward_mask(features, proposals)
                losses.update(self._forward_maskiou(mask_features, proposals, selected_mask, labels, maskiou_targets))
            else:
                losses = self._forward_mask(features, proposals)
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, proposals)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.maskiou_on:
            instances, mask_features = self._forward_mask(features, instances)
            instances = self._forward_maskiou(mask_features, instances)
        else:
            instances = self._forward_mask(features, instances)

        instances = self._forward_keypoint(features, instances)

        return instances


    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            # proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposals, self.training)
            mask_logits = self.mask_head(mask_features)
            if self.maskiou_on:
                loss, selected_mask, labels, maskiou_targets = mask_rcnn_loss(mask_logits, proposals, self.maskiou_on)
                return {"loss_mask": loss}, mask_features, selected_mask, labels, maskiou_targets
            else:
                return {"loss_mask": mask_rcnn_loss(mask_logits, proposals, self.maskiou_on)}
        else:
            # pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, instances)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)

            if self.maskiou_on:
                return instances, mask_features
            else:
                return instances


    def _forward_maskiou(self, mask_features, instances, selected_mask=None, labels=None, maskiou_targets=None):
        """
        Forward logic of the mask iou prediction branch.
        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, calibrate instances' scores.
        """
        if not self.maskiou_on:
            return {} if self.training else instances

        if self.training:
            pred_maskiou = self.maskiou_head(mask_features, selected_mask)
            return {"loss_maskiou": mask_iou_loss(labels, pred_maskiou, maskiou_targets, self.maskiou_weight)}

        else:
            selected_mask = torch.cat([i.pred_masks for i in instances], 0)
            if selected_mask.shape[0] == 0:
                return instances
            pred_maskiou = self.maskiou_head(mask_features, selected_mask)
            mask_iou_inference(instances, pred_maskiou)
            return instances


    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.kp_in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            # proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposals, self.training)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            # pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, instances)
            return self.keypoint_head(keypoint_features, instances)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        weight_init.c2_msra_fill(self.conv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = Max(x)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)



class SpatialAttentionMaskHead(nn.Module):
    """
    A mask head with several conv layers and spatial attention module 
    in CenterMask paper, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(SpatialAttentionMaskHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.spatialAtt = SpatialAttention()

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = self.spatialAtt(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


class CenterMaskHead(Head):
    def __init__(self):
        ...


class CenterMaskPretrained(Model):
    config_class = CenterMaskConfig
    base_model_prefix = 'centermask'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = CenterMaskModel(config)
        # model.state_dict(torch.load('test.pth'))
        return model


class CenterMaskModel(CenterMaskPretrained):
    """
    """
    model_name = 'centermaskv1'

    def __init__(self):
        ...



