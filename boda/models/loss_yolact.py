from collections import defaultdict
from typing import Tuple, List, Dict, Union
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import LossFunction
from ..utils.bbox import match, log_sum_exp, center_size, crop, elemwise_box_iou, jaccard, cxcywh_to_xyxy
from ..utils.mask import elemwise_mask_iou


class Matcher:
    def __init__(
        self, 
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
        true_crowd_boxes,
    ) -> Tuple[Tensor]:
        decoded_priors = self.decode(
            pred_boxes, cxcywh_to_xyxy(pred_priors))

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
        boxes = torch.cat((
            priors[:, :2] + boxes[:, :2] * self.variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(boxes[:, 2:] * self.variances[1])), dim=1)

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    
        return boxes
    
    def encode(self, matched, priors):
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
    return 1 / (1 + np.exp(-x))


class YolactLoss(LossFunction):
    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super().__init__()
        self.num_classes = num_classes + 1
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio
        self.bbox_alpha = 1.5
        self.mask_alpha = 0.4 / 256 * 140 * 140
        self.score_alpha = 1.0

        self.mask_to_train = 300
        self.class_existence_alpha = 1
        self.semantic_segmentation_alpha = 1

        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

    def forward(
        self,
        inputs: Dict[str, List[Tensor]],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Arguments:
            inputs (): 
            targets ():
        Return:
            Dict[str, Tensor]
        """
        self.check_targets(targets)
        targets = self.copy_targets(targets)

        pred_boxes = inputs['boxes']
        pred_scores = inputs['scores']
        pred_masks = inputs['masks']
        pred_priors = inputs['priors']
        pred_prototypes = inputs['proto']  # TODO: proto? prototypes?

        batch_size = len(targets)
        num_priors = pred_priors.size(0)
        
        true_labels = [None] * len(targets)

        matched_pred_boxes = pred_boxes.new_tensor(batch_size, num_priors, 4)
        matched_true_boxes = pred_boxes.new_tensor(batch_size, num_priors, 4)
        matched_pred_scores = pred_boxes.new_tensor(batch_size, num_priors, dtype=torch.int64)
        matched_indexes = pred_boxes.new_tensor(batch_size, num_priors, dtype=torch.int64)

        for i, target in enumerate(targets):
            true_boxes = target['boxes']
            true_labels[i] = target['labels']

            crowds = target['crowds']
            if crowds > 0:
                true_crowd_boxes = true_boxes[-crowds:]
                true_boxes = true_boxes[:-crowds]
                true_labels = true_labels[i][:-crowds]
                true_masks = target['masks'][:-crowds]
            else:
                true_crowd_boxes = None

            matched_boxes, matched_scores, matched_index = Matcher()(
                pred_boxes, pred_priors, true_boxes, true_labels,
                true_crowd_boxes)

            matched_pred_boxes[i] = matched_boxes  # [num_priors,4] encoded offsets to learn
            matched_true_boxes[i, :, :] = true_boxes[matched_indexes[i]]
            matched_pred_scores[i] = matched_scores  # [num_priors] top class label for each prior
            matched_indexes[i] = matched_index  # [num_priors] indices for lookup

        matched_pred_boxes.required_grad = False
        matched_pred_scores.required_grad = False
        matched_indexes.required_grad = False

        pos_scores = matched_pred_scores > 0
        num_pos_scores = pos_scores.sum(dim=1, keepdim=True)
        
        # Size([batch, num_priors, 4])
        pos_index = pos_scores.unsqueeze(pos_scores.dim()).expand_as(pred_boxes)
        
        losses = defaultdict()

        # Localization Loss (Smooth L1)

        pred_boxes = pred_boxes[pos_index].view(-1, 4)
        matched_pred_boxes = matched_pred_boxes[pos_index].view(-1, 4)
        losses['loss_boxes'] = F.smooth_l1_loss(pred_boxes, matched_pred_boxes, reduction='sum') * self.bbox_alpha

        losses['loss_masks'] = self.lincomb_mask_loss(
            pos, idx_t, loc_data, mask_data, priors, proto_data,
            masks, gt_box_t, score_data, inst_data, labels)

        # Confidence loss
        losses['loss_conf'] = self.ohem_conf_loss(conf_data, conf_t, pos, batch_size)

        # Mask IoU Loss
        losses['loss_mask_iou'] = self.mask_iou_loss(net, maskiou_targets)

        # These losses also don't depend on anchors
        losses['loss_semantic'] = self.semantic_segmentation_loss(predictions['segm'], masks, labels)

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        total_num_pos = num_pos.data.sum().float()
        for k in losses:
            if k not in ('P', 'E', 'S'):
                losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size

        return losses

    def lincomb_mask_loss(
        self,
        pos,
        idx_t,
        loc_data,
        mask_data,
        priors,
        proto_data,
        masks,
        gt_box_t,
        score_data,
        inst_data,
        labels,
        interpolation_mode='bilinear'
    ) -> None:
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)

        process_gt_bboxes = self.mask_proto_normalize_emulate_roi_pooling or self.mask_proto_crop

        loss_m = 0
        loss_d = 0 # Coefficient diversity loss

        maskiou_t_list = []
        maskiou_net_input_list = []
        label_t_list = []

        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

                if self.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()

            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            
            if process_gt_bboxes:
                pos_gt_box_t = gt_box_t[idx, cur_pos]

            if pos_idx_t.size(0) == 0:
                continue

            proto_masks = proto_data[idx]
            proto_coef  = mask_data[idx, cur_pos, :]
             
            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.size(0)
            if old_num_pos > self.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[:self.masks_to_train]

                proto_coef = proto_coef[select, :]
                pos_idx_t  = pos_idx_t[select]
                
                if process_gt_bboxes:
                    pos_gt_box_t = pos_gt_box_t[select, :]

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]     
            label_t = labels[idx][pos_idx_t]

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ proto_coef.t()
            # pred_masks = cfg.mask_proto_mask_activation(pred_masks)

            if self.mask_proto_crop:
                pred_masks = crop(pred_masks, pos_gt_box_t)
            
            pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')
            
            if self.mask_proto_normalize_emulate_roi_pooling:
                weight = mask_h * mask_w if cfg.mask_proto_crop else 1
                pos_gt_csize = center_size(pos_gt_box_t)
                gt_box_width  = pos_gt_csize[:, 2] * mask_w
                gt_box_height = pos_gt_csize[:, 3] * mask_h
                pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight

            # If the number of masks were limited scale the loss accordingly
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos

            loss_m += torch.sum(pre_loss)

        losses = {'M': loss_m * self.mask_alpha / mask_h / mask_w}
        
        return losses

    def ohem_conf_loss(self, conf_data, conf_t, pos, num):
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # i.e. -softmax(class 0 confidence)
        loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos]        = 0 # filter out pos boxes
        loss_c[conf_t < 0] = 0 # filter out neutrals (conf_t = -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[pos] = 0
        neg[conf_t < 0] = 0 # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='none')

        loss_c = loss_c.sum()

        return self.conf_alpha * loss_c

    def _mask_iou(self, mask1, mask2):
        intersection = torch.sum(mask1*mask2, dim=(0, 1))
        area1 = torch.sum(mask1, dim=(0, 1))
        area2 = torch.sum(mask2, dim=(0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def mask_iou_loss(self, net, maskiou_targets):
        maskiou_net_input, maskiou_t, label_t = maskiou_targets

        maskiou_p = net.maskiou_net(maskiou_net_input)

        label_t = label_t[:, None]
        maskiou_p = torch.gather(maskiou_p, dim=1, index=label_t).view(-1)

        loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='sum')
        
        return loss_i * self.maskiou_alpha

    def class_existence_loss(self, class_data, class_existence_t):
        return self.class_existence_alpha * F.binary_cross_entropy_with_logits(class_data, class_existence_t, reduction='sum')

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, interpolation_mode='bilinear'):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        loss_s = 0
        
        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]

            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_t[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                
                # Construct Semantic Segmentation
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(segment_t[cur_class_t[obj_idx]], downsampled_masks[obj_idx])
            
            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')

        return loss_s / mask_h / mask_w * self.semantic_segmentation_alpha


