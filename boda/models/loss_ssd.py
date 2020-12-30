from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import LossFunction
from ..utils.bbox import jaccard, cxcywh_to_xyxy
from ..utils.loss import log_sum_exp


class Matcher:
    """Matcher for SSD

    Arguments:
        threshold (float):
        variances (List[float]):
    """
    def __init__(
        self,
        threshold: float = 0.5,
        variances: List[float] = [0.1, 0.2]):
        self.threshold = threshold
        self.variances = variances

    def __call__(
        self,
        pred_boxes,
        pred_scores,
        pred_priors,
        true_boxes,
    ) -> Tuple[Tensor]:
        """
        Arguments:
            pred_boxes (Tensor): Size([N, ])
            pred_priors (Tensor): default boxes Size([N, 4])
            true_boxes (Tensor): ground truth of bounding boxes Size([N, 4])

        Returns:
            matched_boxes (Tensor): Size([num_priors, 4])
            matched_scores (Tensor): Size([num_priors])
        """
        overlaps = jaccard(
            true_boxes, cxcywh_to_xyxy(pred_priors))

        # Best prior for each ground truth 
        best_prior_overlaps, best_prior_indexes = overlaps.max(1, keepdim=True)
        best_prior_indexes.squeeze_(1)
        best_prior_overlaps.squeeze_(1)

        # Best ground truth for each prior boxes (default boxes)
        best_truth_overlaps, best_truth_indexes = overlaps.max(0, keepdim=True)
        best_truth_indexes.squeeze_(0)
        best_truth_overlaps.squeeze_(0)
        best_truth_overlaps.index_fill_(0, best_prior_indexes, 2)

        # TODO refactor: index  best_prior_idx with long tensor
        # Ensure every gt matches with its prior of max overlap
        for j in range(best_prior_indexes.size(0)):
            best_truth_indexes[best_prior_indexes[j]] = j

        matched_boxes = true_boxes[best_truth_indexes]  # Size([N, 4])
        matched_scores = pred_scores[best_truth_indexes] + 1  # Size([N])
        matched_scores[best_truth_overlaps < self.threshold] = 0  # Size([])
        matched_boxes = self.encode(matched_boxes, pred_priors)

        return matched_boxes, matched_scores

    def encode(self, matched_boxes, pred_priors):
        """
        Return:
            (Tensor): Size([num_priors, 4])
        """
        gcxcy = (matched_boxes[:, :2] + matched_boxes[:, 2:])/2 - pred_priors[:, :2]
        gcxcy /= (self.variances[0] * pred_priors[:, 2:])
        gwh = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / pred_priors[:, 2:]
        gwh = torch.log(gwh) / self.variances[1]
        return torch.cat([gcxcy, gwh], dim=1)

    def decode(self, pred_boxes, pred_priors):
        boxes = torch.cat((
            pred_priors[:, :2] + pred_boxes[:, :2] * self.variances[0] * pred_priors[:, 2:],
            pred_priors[:, 2:] * torch.exp(pred_boxes[:, 2:] * self.variances[1])), dim=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes


class SsdLoss(LossFunction):
    def __init__(
        self,
        size,
        overlap_thresh,
        prior_for_matching,
        bkg_label,
        neg_mining,
        neg_pos,
        neg_overlap,
        encode_target,
        variances: List[float] = [0.1, 0.2]
    ) -> None:
        super().__init__()
        self.num_classes = config.num_classes + 1
        self.variances = variances
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap

    def forward(self, inputs, targets):
        """
        """
        self.check_targets(targets)
        targets = self.copy_targets(targets)

        pred_boxes = inputs['boxes']
        num_boxes = pred_boxes.size(0)
        pred_scores = inputs['scores']
        pred_priors = inputs['priors']
        pred_priors = pred_priors[:pred_boxes.size(1), :]

        batch_size = len(targets)
        num_priors = pred_priors.size(0)

        # match priors (default boxes) and ground truth boxes
        matched_true_boxes = pred_boxes.new_tensor(batch_size, num_priors, 4)
        matched_true_scores = pred_boxes.new_tensor(batch_size, num_priors, dtype=torch.int64)

        for i, target in enumerate(targets):
            true_boxes = target['boxes']
            true_labels = target['labels']
            matched_boxes, matched_scores = Matcher(self.threshold)(
                pred_boxes, pred_priors, true_boxes, true_labels)

            matched_true_boxes[i] = matched_boxes
            matched_true_scores[i] = matched_scores

        matched_true_boxes.requires_grad = False
        matched_true_scores.requires_grad = False

        # TODO: positive_scores or pos_scores
        pos = matched_true_scores > 0
        num_pred_scores = pos.sum(dim=1, keepdim=True)

        pos_indexes = pos.unsqueeze(pos.dim()).expand_as(pred_boxes)
        matched_pred_boxes = pred_boxes[pos_indexes].view(-1, 4)
        matched_true_boxes = matched_true_boxes[pos_indexes].view(-1, 4)

        loss_box = F.smooth_l1_loss(
            matched_pred_boxes, matched_true_boxes, size_average=False)

        # Compute hard negative mining
        pred_scores = pred_scores.view(-1, self.num_classes)
        loss_score = log_sum_exp(pred_scores) - pred_scores.gather(1, matched_true_scores.view(-1, 1))

        # Hard negative mining
        loss_score[pos] = 0
        loss_score = loss_score.view(num_boxes, -1)

        _, loss_index = loss_score.sort(1, descending=True)
        _, rank_index = loss_index.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = rank_index < num_neg.expand_as(rank_index)

        # Confidence loss including positive and negative samples
        pos_index = pos.unsqueeze(2).expand_as(pred_scores)
        neg_index = neg.unsqueeze(2).expand_as(pred_scores)

        pred_scores = pred_boxes[(pos_index + neg_index).gt(0)].view(-1, self.num_classes)
        weighted_targets = matched_true_scores[(pos+neg).gt(0)]
        loss_score = F.cross_entropy(pred_scores, weighted_targets, size_average=False)

        losses = {
            'loss_bbox': None,
            'loss_conf': None,
        }

        return losses