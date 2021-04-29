import os
from typing import List, Union, Sequence

import torch
from torch import nn, Tensor
from ...base_architecture import Neck, Head, Model
from ..neck_fpn import FeaturePyramidNetworks
from .configuration_fcos import FcosConfig
from ..backbone_resnet import resnet101


class FcosPredictNeck(FeaturePyramidNetworks):
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
                    padding=1, bias=True
                ),
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
                    padding=1, bias=True
                ),
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
                    padding=1, bias=True
                ),
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

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def forward(self, inputs):
        inputs = self.check_inputs(inputs)
        inputs = self.resize_inputs(inputs)
        outputs = self.backbone(inputs)
        outputs = [outputs[i] for i in self.selected_layers]
        outputs = self.neck(outputs)
        o1, o2, o3 = self.heads(outputs)
        if self.training:
            pass
        else:
            sampled_boxes = []
            bundle = (
                self.locations, self.logits_pred,
                self.reg_pred, self.ctrness_pred,
                self.strides
            )

            for i, (l, o, r, c, s) in enumerate(zip(*bundle)):
                # recall that during training, we normalize regression targets with FPN's stride.
                # we denormalize them here.
                r = r * s
                sampled_boxes.append(
                    self.forward_for_single_feature_map(
                        l, o, r, c, self.image_sizes
                    )
                )

            boxlists = list(zip(*sampled_boxes))
            boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
            boxlists = self.select_over_all_levels(boxlists)

            return outputs

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

    def forward_for_single_feature_map(
            self, locations, box_cls,
            reg_pred, ctrness, image_sizes
    ):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness = ctrness.reshape(N, -1).sigmoid()

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]
        candidate_inds = box_cls > self.pre_nms_thresh
        # pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        if not self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            # per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_candidate_nonzeros = torch.nonzero(per_candidate_inds, as_tuple=False)
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations

            results.append(boxlist)

        return results
