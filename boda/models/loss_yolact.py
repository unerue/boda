import sys
from collections import defaultdict
from typing import Tuple, List, Dict, Union
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import LossFunction
from ..utils.bbox import elemwise_box_iou, jaccard, cxcywh_to_xyxy, crop, xyxy_to_cxywh
from ..utils.loss import log_sum_exp
from ..utils.mask import elemwise_mask_iou


class Matcher:
    """

    Args:
        pos_threshold ():
        ? positive_threshold ():
        neg_threshold ():
        ? negative_threshold ():
        crowd_iou_threshold ():
        variances ():
    """
    def __init__(
        self,
        positive_margin: float = 0.5,
        positive_threshold: float = 0.5,
        negative_threshold: float = 0.5,
        pos_thresh: float = 0.5,
        neg_thresh: float = 0.5,
        crowd_iou_thresh: int = 1,
        variances: List[int] = [0.1, 0.2]
    ) -> None:
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.crowd_iou_thresh = crowd_iou_thresh
        self.variances = variances

    def __call__(
        self,
        pred_boxes,
        pred_priors,
        true_boxes,
        true_labels,
        true_crowds,
    ) -> Tuple[Tensor]:
        """
        Args:
            pred_boxes ():
            pred_priors ():
            true_boxes ():
            true_labels ():
            true_crowds ():

        Returns:
            boxes (:obj:`FloatTensor[N, 4]`): N is a number of prior boxes 
            scores (:obj:`LongTensor[N]`):
            best_truth_index (:obj:`LongTensor[N]`):
        """
        # FloatTensor[N, 4]
        decoded_priors = self.decode(
            pred_boxes, pred_priors)

        # LongTensor[number of ground truths]
        overlaps = jaccard(true_boxes, decoded_priors)
        best_truth_overlap, best_truth_index = overlaps.max(0)

        for _ in range(overlaps.size(0)):
            best_prior_overlap, best_prior_index = overlaps.max(1)
            j = best_prior_overlap.max(0)[1]
            i = best_prior_index[j]

            overlaps[:, i] = -1
            overlaps[j, :] = -1

            best_truth_overlap[i] = 2
            best_truth_index[i] = j

        matches = true_boxes[best_truth_index]  # Size([num_priors,4])
        scores = true_labels[best_truth_index] + 1  # Size([num_priors])

        scores[best_truth_overlap < self.pos_thresh] = -1  # label as neutral
        scores[best_truth_overlap < self.neg_thresh] = 0  # label as background

        # Deal with crowd annotations for COCO
        # if crowd_boxes is not None and self.crowd_iou_threshold < 1:
        #     # Size [num_priors, num_crowds]
        #     crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        #     # Size [num_priors]
        #     best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        #     # Set non-positives with crowd iou of over the threshold to be neutral.
        #     conf[(conf <= 0) & (best_crowd_overlap > self.crowd_iou_threshold)] = -1

        boxes = self.encode(matches, pred_priors)

        return boxes, scores, best_truth_index

    def decode(self, boxes, priors):
        """
        Args:
            boxes (FloatTensor[N, 4]): N is the number of prior boxes
            priors (FloatTensor[N, 4]):
        Return:
            boxes (FloatTensor[N, 4]):
        """
        boxes = torch.cat((
            priors[:, :2] + boxes[:, :2] * self.variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(boxes[:, 2:] * self.variances[1])), dim=1)

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    
        return boxes

    def encode(self, matched, priors):
        """
        Args:
            matched ()
            priors ()

        Return:
            boxes ():
        """
        # dist b/t match center and prior's center
        gcxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
        # encode variance
        gcxcy /= (self.variances[0] * priors[:, 2:])
        # match wh / prior wh
        gwh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        gwh = torch.log(gwh) / self.variances[1]
        # return target for smooth_l1_loss
        boxes = torch.cat([gcxcy, gwh], dim=1)  # [num_priors,4]

        return boxes


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class YolactLoss(LossFunction):
    """Loss Function for YOLACT

    Arguments:
        num_classes (int):
        pos_threshold (float)
    """
    def __init__(
        self,
        positive_threshold: float = 0.5,
        negative_threshold: float = 0.4,
        neg_pos_ratio: float = 1.0,
        masks_to_train: int = 100
    ) -> None:
        super().__init__()
        self.pos_threshold = positive_threshold
        self.neg_threshold = negative_threshold
        self.negpos_ratio = neg_pos_ratio
        self.bbox_weight = 1.5
        self.conf_weight = 1.0
        self.score_weight = 1.0
        self.confidence_weight = 1.5
        self.mask_weight = 1.5

        self.crowd_iou_threshold = 0.7

        self.bbox_alpha = 1.5
        self.mask_alpha = 6.125
        self.score_alpha = 1.0
        self.conf_alpha = 1.0
        self.semantic_segmentation_alpha = 1.0
        self.class_existence_alpha = 1.0

        self.masks_to_train = masks_to_train

        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

    def forward(
        self,
        inputs: Dict[str, List[Tensor]],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Args:
            inputs (Dict[str, List[Tensor]]):
                `boxes` (:obj:`FloatTensor[B, N*S, 4]`): B is the number of batch size, S is the number of selected_layers
                `masks` (:obj:`FloatTensor[B, N*S, P]`): P is the number of prototypes
                `scores` (:obj:`FloatTensor[B, N*S, C]`): C is the number of classes with background e.g. 80 + 1
                `priors` (:obj:`FloatTensor[N, 4]`):
                `prototype_masks` (:obj:`FloatTensor[B, H, W, P]`):
                `semantic_masks` (:obj:`FloatTensor[B, C, H, W]`)::

            targets (List[Dict[str, Tensor]]):
                `boxes` (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2]
                `masks` (ByteTensor[N, H, W]): the segmentation binary masks for each instance
                `labels` (LongTensor[N]): the class label for each ground-truth
                `crowds` (LongTensor[N]):
                `areas` (FloatTensor[N]):

        Returns:
            return_dict (:obj:`Dict[str, Tensor]`):
                'boxes` ()
        """
        self.check_targets(targets)
        targets = self.copy_targets(targets)

        losses = defaultdict()

        pred_boxes = inputs['boxes']
        pred_masks = inputs['masks']
        pred_scores = inputs['scores']
        pred_priors = inputs['priors']
        pred_prototype_masks = inputs['prototype_masks']
        pred_semantic_masks = inputs['semantic_masks']

        batch_size = len(targets)
        num_priors = pred_priors.size(0)
        num_classes = pred_scores.size(2)
        
        matched_pred_boxes = pred_boxes.new(batch_size, num_priors, 4)
        matched_true_boxes = pred_boxes.new(batch_size, num_priors, 4)
        matched_pred_scores = pred_boxes.new(batch_size, num_priors)
        matched_indexes = pred_boxes.new(batch_size, num_priors).long()

        true_masks = []
        true_labels = []
        for i, target in enumerate(targets):
            true_boxes = target['boxes']
            true_boxes /= torch.as_tensor([550, 550, 550, 550], dtype=torch.float32, device=pred_boxes.device)

            true_masks.append(target['masks'])
            true_labels.append(target['labels'])
            # true_labels[i] = target['labels']
            true_crowds = target['crowds']

            # if true_crowds > 0:
            #     true_crowd_boxes = true_boxes[-crowds:]
            #     true_boxes = true_boxes[:-crowds]
            #     true_labels = true_labels[i][:-crowds]
            #     true_masks = target['masks'][:-crowds]
            # else:
            #     true_crowd_boxes = None

            matched_boxes, matched_scores, matched_index = Matcher()(
                pred_boxes[i],
                pred_priors,
                true_boxes,
                true_labels[i],
                true_crowds)

            matched_pred_boxes[i] = matched_boxes  # [num_priors,4] encoded offsets to learn
            matched_pred_scores[i] = matched_scores  # [num_priors] top class label for each prior
            matched_indexes[i] = matched_index
            matched_true_boxes[i, :, :] = true_boxes[matched_indexes[i]]
        
        matched_pred_boxes.required_grad = False
        matched_pred_scores.required_grad = False
        matched_indexes.required_grad = False
    
        positive_scores = matched_pred_scores > 0
        num_positive_scores = positive_scores.sum(dim=1, keepdim=True)

        # Size([batch, num_priors, 4])
        positive_index = positive_scores.unsqueeze(positive_scores.dim()).expand_as(pred_boxes)
    
        pred_boxes = pred_boxes[positive_index].view(-1, 4)
        matched_pred_boxes = matched_pred_boxes[positive_index].view(-1, 4)

        # Localization loss (Smooth L1)
        losses['B'] = \
            F.smooth_l1_loss(
                pred_boxes,
                matched_pred_boxes, reduction='sum') * self.bbox_alpha

        losses['M'] = self.lincomb_mask_loss(
            positive_scores,
            matched_indexes,
            pred_masks,
            pred_prototype_masks,
            true_masks,
            matched_true_boxes)

        # Confidence loss
        losses['C'] = self.ohem_conf_loss(
            pred_scores,
            matched_pred_scores,
            positive_scores,
            batch_size,
            num_classes)

        losses['S'] = self.semantic_segmentation_loss(
            pred_semantic_masks,
            true_masks,
            true_labels)

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        total_num_positives = num_positive_scores.data.sum().float()
        for k in losses:
            # if k not in ('P', 'E', 'S'):
            if k not in ('S', ):
                losses[k] /= total_num_positives
            else:
                losses[k] /= batch_size

        return losses

    def lincomb_mask_loss(
        self,
        positive_scores,
        matched_indexes,
        pred_masks,
        prototype_masks,
        true_masks,
        matched_true_boxes,
        mode: str = 'bilinear'
    ) -> Tensor:
        """
        Args:
            mode (:obj:`str`): interpolation mode
        Returns:
        """
        h = prototype_masks.size(1)
        w = prototype_masks.size(2)

        loss = 0
        for i in range(pred_masks.size(0)):
            with torch.no_grad():
                downsampled_masks = F.interpolate(
                    true_masks[i].unsqueeze(0).float(), (h, w),
                    mode=mode, align_corners=False
                ).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
                # mask_proto_binarize_downsampled_gt
                downsampled_masks = downsampled_masks.gt(0.5).float()

            j = positive_scores[i]  # current position
            positive_index = matched_indexes[i, j]  # positive index
            positive_true_boxes = matched_true_boxes[i, j]  # process_gt_boxes

            if positive_index.size(0) == 0:
                continue

            proto_masks = prototype_masks[i]
            proto_coef = pred_masks[i, j, :]

            # If we have over the allowed number of masks, select a random sample
            previous_num_positives = proto_coef.size(0)
            if previous_num_positives > self.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                selected_masks = perm[:self.masks_to_train]  # select masks to train

                proto_coef = proto_coef[selected_masks, :]
                positive_index  = positive_index[selected_masks]
                
                positive_true_boxes = positive_true_boxes[selected_masks, :]  # process_gt_boxes

            num_positives = proto_coef.size(0)
            _true_masks = downsampled_masks[:, :, positive_index]
            # Size([h, w, num_positives])
            _pred_masks = proto_masks @ proto_coef.t()
            _pred_masks = torch.sigmoid(_pred_masks)
            _pred_masks = crop(_pred_masks, positive_true_boxes)

            _loss = F.binary_cross_entropy(
                torch.clamp(_pred_masks, 0, 1), _true_masks, reduction='none')

            # mask_proto_normalize_emulate_roi_pooling
            weight = h * w
            positive_true_csize = xyxy_to_cxywh(positive_true_boxes)
            true_boxes_width  = positive_true_csize[:, 2] * w
            true_boxes_height = positive_true_csize[:, 3] * h

            _loss = _loss.sum(dim=(0, 1)) / true_boxes_width / true_boxes_height * weight

            # If the number of masks were limited scale the loss accordingly
            if previous_num_positives > num_positives:
                _loss *= previous_num_positives / num_positives

            loss += torch.sum(_loss)

        return loss * self.mask_alpha / h / w

    def ohem_conf_loss(
        self,
        pred_scores,
        matched_pred_scores,
        positive_scores,
        batch_size,
        num_classes
    ) -> Tensor:
        # Compute max conf across batch for hard negative mining
        batch_scores = pred_scores.view(-1, num_classes)
        # i.e. -softmax(class 0 confidence)
        # TODO: remove squeeze(1)
        # batch_scores Size([N, C]) -> log_sum_exp(batch_scores) Size([N, 1])
        loss = log_sum_exp(batch_scores).squeeze(1) - batch_scores[:, 0]
        # Hard Negative Mining
        loss = loss.view(batch_size, -1)

        loss[positive_scores] = 0  # filter out pos boxes
        loss[matched_pred_scores < 0] = 0  # filter out neutrals (conf_t = -1)

        _, loss_index = loss.sort(1, descending=True)
        _, ranked_index = loss_index.sort(1)
        num_positives = positive_scores.long().sum(1, keepdim=True)
        num_negatives = torch.clamp(self.negpos_ratio * num_positives, max=positive_scores.size(1)-1)
        negatives = ranked_index < num_negatives.expand_as(ranked_index)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        negatives[positive_scores] = 0
        negatives[matched_pred_scores < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        positive_index = positive_scores.unsqueeze(2).expand_as(pred_scores)
        negative_index = negatives.unsqueeze(2).expand_as(pred_scores)
        conf_p = pred_scores[(positive_index + negative_index).gt(0)].view(-1, num_classes)
        targets_weighted = matched_pred_scores[(positive_scores+negatives).gt(0)].long()
        loss = F.cross_entropy(conf_p, targets_weighted, reduction='none')

        return self.conf_alpha * loss.sum()

    def semantic_segmentation_loss(
        self,
        segmentic_masks,
        true_masks,
        true_labels,
        mode='bilinear'
    ) -> Tensor:
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, _, h, w = segmentic_masks.size()
        loss = 0
        for i in range(batch_size):
            pred_segmentic_mask = segmentic_masks[i]
            true_label = true_labels[i]
            with torch.no_grad():
                downsampled_masks = F.interpolate(
                    true_masks[i].unsqueeze(0).float(), (h, w),
                    mode=mode, align_corners=False
                ).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                # Construct semantic segmentation
                true_segmentic_mask = \
                    torch.zeros_like(pred_segmentic_mask, requires_grad=False)
                for j in range(downsampled_masks.size(0)):
                    # print(true_label, j)
                    true_segmentic_mask[true_label[j]] = \
                        torch.max(
                            true_segmentic_mask[true_label[j]],
                            downsampled_masks[j])

            loss += F.binary_cross_entropy_with_logits(
                pred_segmentic_mask, true_segmentic_mask, reduction='sum')

        return loss / h / w * self.semantic_segmentation_alpha



