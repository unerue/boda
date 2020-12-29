import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import LossFunction
from ..utils.bbox import jaccard, cxcywh_to_xyxy
from ..utils.loss import log_sum_exp


class Matcher:
    def __init__(self, config, threshold, variances):
        self.config = config
        self.threshold = threshold
        self.variances = variances

    def __call__(
        self,
        pred_boxes,
        pred_priors,
        true_boxes,
        idx
    ) -> Tuple[Tensor]:
        overlaps = jaccard(
            true_boxes,
            cxcywh_to_xyxy(pred_priors))

        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
        # [1,num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
        # TODO refactor: index  best_prior_idx with long tensor
        # ensure every gt matches with its prior of max overlap
        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j

        matches = truths[best_truth_idx]          # Shape: [num_priors,4]
        conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
        conf[best_truth_overlap < threshold] = 0  # label as background
        loc = self.encode(matches, priors, variances)

        return loc, conf
        
    def encode(self, matched, priors, variances):
        gcxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
        gcxcy /= (variances[0] * priors[:, 2:])
        gwh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        gwh = torch.log(gwh) / variances[1]
        return torch.cat([gcxcy, gwh], 1)  # [num_priors,4]

    def decode(self, loc, priors, variances):
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), dim=1)
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
    ):
        super().__init__()
        self.num_classes = config.num_classes + 1
        self.variances = config.variances
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
        pred_scores = inputs['scores']
        pred_priors = inputs['priors']

        true_boxes = targets['boxes']
        true_labels = targets['labels']

        loc_data, conf_data, priors = inputs

        # batch_size
        bs = len(targets)
        priors = pred_priors[:pred_boxes.size(1), :]
        num_priors = priors.size(0)

        # match priors (default boxes) and ground truth boxes
        default_boxes = torch.zeros(bs, num_priors, 4)
        default_scores = torch.zeros(bs, num_priors)

        matched_pred_boxes = pred_boxes.new_tensor(batch_size, num_priors, 4)
        matched_pred_scores = pred_boxes.new_tensor(batch_size, num_priors, dtype=torch.int64)
        

        for i in range(bs):
            _true_boxes = true_boxes[i]
            _true_labels = true_labels[i]
            Match(self.threshold)(
                _true_boxes, default_boxes, variance,
                _true_labels, default_boxes, default_scores, i)
        
        loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
        conf_t[idx] = conf  # [num_priors] top class label for each prior

        default_boxes.requires_grad =False
        default_scores.requires_grad = False

        pos = default_scores > 0

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(pred_boxes)
        _pred_boxes = pred_boxes[pos_idx].view(-1, 4)
        _default_boxes = default_boxes[pos_idx].view(-1, 4)


        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) / N

        batch_scores = pred_scores.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_scores) - batch_scores.gather(1, default_scores.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_class = F.cross_entropy(conf_p, targets_weighted, size_average=False) / N

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        losses = {
            'loss_boxes': loss_boxes,
            'loss_class': loss_class,
        }
        return losses

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        # num_pos = pos.sum(keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c