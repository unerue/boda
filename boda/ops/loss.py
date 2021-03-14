import math
import numpy as np
from numpy import ndarray

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def sigmoid(x: ndarray) -> ndarray:
    """Sigmoid for NumPy"""
    return 1 / (1 + np.exp(-x))


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def lincomb_mask_loss(
    positive_scores,
    matched_indexes,
    pred_masks,
    pred_proto_masks,
    true_masks,
    matched_true_boxes,
    mode: str = 'bilinear',
    masks_to_train: int = 200,
) -> Tensor:
    """
    Args:
        mode (:obj:`str`): interpolation mode

    Returns:
    """
    h = pred_proto_masks.size(1)
    w = pred_proto_masks.size(2)

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

        proto_masks = pred_proto_masks[i]
        proto_coef = pred_masks[i, j, :]

        # If we have over the allowed number of masks, select a random sample
        previous_num_positives = proto_coef.size(0)
        if previous_num_positives > masks_to_train:
            perm = torch.randperm(proto_coef.size(0))
            selected_masks = perm[:masks_to_train]  # select masks to train

            proto_coef = proto_coef[selected_masks, :]
            positive_index = positive_index[selected_masks]

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
        true_boxes_width = positive_true_csize[:, 2] * w
        true_boxes_height = positive_true_csize[:, 3] * h

        _loss = _loss.sum(dim=(0, 1)) / true_boxes_width / true_boxes_height * weight

        # If the number of masks were limited scale the loss accordingly
        if previous_num_positives > num_positives:
            _loss *= previous_num_positives / num_positives

        loss += torch.sum(_loss)

    return loss / h / w


def ohem_conf_loss(
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

    return loss.sum()


def semantic_mask_loss(
    pred_masks,
    true_masks,
    true_labels,
    mode='bilinear'
) -> Tensor:
    # Note num_classes here is without the background class so cfg.num_classes-1
    batch_size, _, h, w = pred_masks.size()
    loss = 0
    for i in range(batch_size):
        pred_segmentic_mask = pred_masks[i]
        true_label = true_labels[i]
        with torch.no_grad():
            downsampled_masks = F.interpolate(
                true_masks[i].unsqueeze(0).float(), (h, w),
                mode=mode, align_corners=False).squeeze(0)

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

    return loss / h / w


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def focal_conf_loss(
    conf_data,
    conf_t,
    alpha,
    gamma
) -> Tensor:
    """
    Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
    Adapted from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    Note that this uses softmax and not the original sigmoid from the paper.
    """
    conf_t = conf_t.view(-1)  # [batch_size*num_priors]
    conf_data = conf_data.view(-1, conf_data.size(-1))  # [batch_size*num_priors, num_classes]

    # Ignore neutral samples (class < 0)
    keep = (conf_t >= 0).float()
    conf_t[conf_t < 0] = 0  # so that gather doesn't drum up a fuss

    logpt = F.log_softmax(conf_data, dim=-1)
    logpt = logpt.gather(1, conf_t.unsqueeze(-1))
    logpt = logpt.view(-1)
    pt = logpt.exp()

    # I adapted the alpha_t calculation here from
    # https://github.com/pytorch/pytorch/blob/master/modules/detectron/softmax_focal_loss_op.cu
    # You'd think you want all the alphas to sum to one, but in the original implementation they
    # just give background an alpha of 1-alpha and each forground an alpha of alpha.
    background = (conf_t == 0).float()
    at = (1 - alpha) * background + alpha * (1 - background)

    loss = -at * (1 - pt) ** gamma * logpt

    # See comment above for keep
    return cfg.conf_alpha * (loss * keep).sum()


def focal_conf_sigmoid_loss(
    conf_data,
    conf_t):
    """
    Focal loss but using sigmoid like the original paper.
    Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
            Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
    """
    num_classes = conf_data.size(-1)

    conf_t = conf_t.view(-1) # [batch_size*num_priors]
    conf_data = conf_data.view(-1, num_classes) # [batch_size*num_priors, num_classes]

    # Ignore neutral samples (class < 0)
    keep = (conf_t >= 0).float()
    conf_t[conf_t < 0] = 0 # can't mask with -1, so filter that out

    # Compute a one-hot embedding of conf_t
    # From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
    conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
    conf_pm_t  = conf_one_t * 2 - 1 # -1 if background, +1 if forground for specific class

    logpt = F.logsigmoid(conf_data * conf_pm_t) # note: 1 - sigmoid(x) = sigmoid(-x)
    pt    = logpt.exp()

    at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (1 - conf_one_t)
    at[..., 0] = 0 # Set alpha for the background class to 0 because sigmoid focal loss doesn't use it

    loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
    loss = keep * loss.sum(dim=-1)

    return cfg.conf_alpha * loss.sum()


def focal_conf_objectness_loss(
    conf_data,
    conf_t):
    """
    Instead of using softmax, use class[0] to be the objectness score and do sigmoid focal loss on that.
    Then for the rest of the classes, softmax them and apply CE for only the positive examples.
    If class[0] = 1 implies forground and class[0] = 0 implies background then you achieve something
    similar during test-time to softmax by setting class[1:] = softmax(class[1:]) * class[0] and invert class[0].
    """

    conf_t = conf_t.view(-1) # [batch_size*num_priors]
    conf_data = conf_data.view(-1, conf_data.size(-1)) # [batch_size*num_priors, num_classes]

    # Ignore neutral samples (class < 0)
    keep = (conf_t >= 0).float()
    conf_t[conf_t < 0] = 0 # so that gather doesn't drum up a fuss

    background = (conf_t == 0).float()
    at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)

    logpt = F.logsigmoid(conf_data[:, 0]) * (1 - background) + F.logsigmoid(-conf_data[:, 0]) * background
    pt    = logpt.exp()

    obj_loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

    # All that was the objectiveness loss--now time for the class confidence loss
    pos_mask = conf_t > 0
    conf_data_pos = (conf_data[:, 1:])[pos_mask] # Now this has just 80 classes
    conf_t_pos    = conf_t[pos_mask] - 1         # So subtract 1 here

    class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')

    return cfg.conf_alpha * (class_loss + (obj_loss * keep).sum())


def conf_objectness_loss(
    conf_data,
    conf_t,
    batch_size,
    loc_p,
    loc_t,
    priors
):
    """
    Instead of using softmax, use class[0] to be p(obj) * p(IoU) as in YOLO.
    Then for the rest of the classes, softmax them and apply CE for only the positive examples.
    """

    conf_t = conf_t.view(-1) # [batch_size*num_priors]
    conf_data = conf_data.view(-1, conf_data.size(-1)) # [batch_size*num_priors, num_classes]

    pos_mask = (conf_t > 0)
    neg_mask = (conf_t == 0)

    obj_data = conf_data[:, 0]
    obj_data_pos = obj_data[pos_mask]
    obj_data_neg = obj_data[neg_mask]

    # Don't be confused, this is just binary cross entropy similified
    obj_neg_loss = - F.logsigmoid(-obj_data_neg).sum()

    with torch.no_grad():
        pos_priors = priors.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 4)[pos_mask, :]

        boxes_pred = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
        boxes_targ = decode(loc_t, pos_priors, cfg.use_yolo_regressors)

        iou_targets = elemwise_box_iou(boxes_pred, boxes_targ)

    obj_pos_loss = - iou_targets * F.logsigmoid(obj_data_pos) - (1 - iou_targets) * F.logsigmoid(-obj_data_pos)
    obj_pos_loss = obj_pos_loss.sum()

    # All that was the objectiveness loss--now time for the class confidence loss
    conf_data_pos = (conf_data[:, 1:])[pos_mask] # Now this has just 80 classes
    conf_t_pos    = conf_t[pos_mask] - 1         # So subtract 1 here

    class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')

    return cfg.conf_alpha * (class_loss + obj_pos_loss + obj_neg_loss)


def direct_mask_loss(
    pos_idx,
    idx_t,
    loc_data,
    mask_data,
    priors,
    masks
) -> Tensor:
    """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
    loss_m = 0
    for idx in range(mask_data.size(0)):
        with torch.no_grad():
            cur_pos_idx = pos_idx[idx, :, :]
            cur_pos_idx_squeezed = cur_pos_idx[:, 1]

            # Shape: [num_priors, 4], decoded predicted bboxes
            pos_bboxes = decode(loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors)
            pos_bboxes = pos_bboxes[cur_pos_idx].view(-1, 4).clamp(0, 1)
            pos_lookup = idx_t[idx, cur_pos_idx_squeezed]

            cur_masks = masks[idx]
            pos_masks = cur_masks[pos_lookup, :, :]
            
            # Convert bboxes to absolute coordinates
            num_pos, img_height, img_width = pos_masks.size()

            # Take care of all the bad behavior that can be caused by out of bounds coordinates
            x1, x2 = sanitize_coordinates(pos_bboxes[:, 0], pos_bboxes[:, 2], img_width)
            y1, y2 = sanitize_coordinates(pos_bboxes[:, 1], pos_bboxes[:, 3], img_height)

            # Crop each gt mask with the predicted bbox and rescale to the predicted mask size
            # Note that each bounding box crop is a different size so I don't think we can vectorize this
            scaled_masks = []
            for jdx in range(num_pos):
                tmp_mask = pos_masks[jdx, y1[jdx]:y2[jdx], x1[jdx]:x2[jdx]]

                # Restore any dimensions we've left out because our bbox was 1px wide
                while tmp_mask.dim() < 2:
                    tmp_mask = tmp_mask.unsqueeze(0)

                new_mask = F.adaptive_avg_pool2d(tmp_mask.unsqueeze(0), cfg.mask_size)
                scaled_masks.append(new_mask.view(1, -1))

            mask_t = torch.cat(scaled_masks, 0).gt(0.5).float() # Threshold downsampled mask
        
        pos_mask_data = mask_data[idx, cur_pos_idx_squeezed, :]
        loss_m += F.binary_cross_entropy(torch.clamp(pos_mask_data, 0, 1), mask_t, reduction='sum') * cfg.mask_alpha

    return loss_m


def coeff_diversity_loss(self, coeffs, instance_t):
    """
    coeffs     should be size [num_pos, num_coeffs]
    instance_t should be size [num_pos] and be values from 0 to num_instances-1
    """
    num_pos = coeffs.size(0)
    instance_t = instance_t.view(-1) # juuuust to make sure

    coeffs_norm = F.normalize(coeffs, dim=1)
    cos_sim = coeffs_norm @ coeffs_norm.t()

    inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim)).float()

    # Rescale to be between 0 and 1
    cos_sim = (cos_sim + 1) / 2

    # If they're the same instance, use cosine distance, else use cosine similarity
    loss = (1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)

    # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
    # and all the losses will be divided by num_pos at the end, so just one extra time.
    return cfg.mask_proto_coeff_diversity_alpha * loss.sum() / num_pos