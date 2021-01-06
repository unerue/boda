import math
from typing import Tuple, List
import numpy as np

import torch
from torch import nn, Tensor


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Argument:
        boxes (Tensor): [[cx, cy, w, h]] center-size default boxes from priorbox layers.
    Return:
        boxes: (Tensor) Converted [[xmin, ymin, xmax, ymax]] form of boxes.
    """
    return torch.cat(
        (boxes[:, :2]-boxes[:, 2:] / 2, boxes[:, :2]+boxes[:, 2:] / 2), dim=1)


def xyxy_to_cxywh(boxes: Tensor) -> Tensor:
    """Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Argument:
        boxes (Tensor): [xmin, ymin, xmax, ymax] point_form boxes
    Return:
        boxes (Tensor): Converted [cx, cy, w, h] form of boxes.
    """
    return torch.cat(
        ((boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2]), dim=1)


def gcxywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert grid center point to 
    Argument:
        boxes (Tensor): [gcx, gcy, w, h]
    Return:
    """
    xyxy = torch.zeros_like(boxes)
    xyxy[:, :2] = boxes
    return xyxy


def intersect(box_a: Tensor, box_b: Tensor) -> Tensor:
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (Tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(
        box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))

    min_xy = torch.max(
        box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))

    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter


def intersect_numpy(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard(box_a: Tensor, box_b: Tensor, iscrowd: bool = False) -> Tensor:
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = (
        (box_a[:, :, 2]-box_a[:, :, 0]) *
        (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = (
        (box_b[:, :, 2]-box_b[:, :, 0]) *
        (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / union

    return out if use_batch else out.squeeze(0)


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def elemwise_box_iou(box_a, box_b):
    """ Does the same as above but instead of pairwise, elementwise along the inner dimension. """
    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, 0] * inter[:, 1]

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    union = area_a + area_b - inter
    union = torch.clamp(union, min=0.1)

    # Return value is [n] for inputs [n, 4]
    return torch.clamp(inter / union, max=1)



class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes
        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        pred_boxes = self.decode_single(
            rel_codes.reshape(box_sum, -1), concat_boxes
        )
        return pred_boxes.reshape(box_sum, -1, 4)

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes




# def generate_ssd_priors(specs: List, image_size, clamp=True) -> torch.Tensor:
#     """Generate SSD Prior Boxes.
#     It returns the center, height and width of the priors. The values are relative to the image size
#     Args:
#         specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
#             specs = [
#                 SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
#                 SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#                 SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#                 SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#                 SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#                 SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
#             ]
#         image_size: image size.
#         clamp: if true, clamp the values to make fall between [0.0, 1.0]
#     Returns:
#         priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
#             are relative to the image size.
#     """
#     priors = []
#     for spec in specs:
#         scale = image_size / spec.shrinkage
#         for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
#             x_center = (i + 0.5) / scale
#             y_center = (j + 0.5) / scale

#             # small sized square box
#             size = spec.box_sizes.min
#             h = w = size / image_size
#             priors.append([
#                 x_center,
#                 y_center,
#                 w,
#                 h
#             ])

#             # big sized square box
#             size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
#             h = w = size / image_size
#             priors.append([
#                 x_center,
#                 y_center,
#                 w,
#                 h
#             ])

#             # change h/w ratio of the small sized box
#             size = spec.box_sizes.min
#             h = w = size / image_size
#             for ratio in spec.aspect_ratios:
#                 ratio = math.sqrt(ratio)
#                 priors.append([
#                     x_center,
#                     y_center,
#                     w * ratio,
#                     h / ratio
#                 ])
#                 priors.append([
#                     x_center,
#                     y_center,
#                     w / ratio,
#                     h * ratio
#                 ])

#     priors = torch.tensor(priors)
#     if clamp:
#         torch.clamp(priors, 0.0, 1.0, out=priors)
#     return priors


# def assign_priors(gt_boxes, gt_labels, corner_form_priors,
#                   iou_threshold):
#     """Assign ground truth boxes and targets to priors.
#     Args:
#         gt_boxes (num_targets, 4): ground truth boxes.
#         gt_labels (num_targets): labels of targets.
#         priors (num_priors, 4): corner form priors
#     Returns:
#         boxes (num_priors, 4): real values for priors.
#         labels (num_priros): labels for priors.
#     """
#     # size: num_priors x num_targets
#     ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
#     # size: num_priors
#     best_target_per_prior, best_target_per_prior_index = ious.max(1)
#     # size: num_targets
#     best_prior_per_target, best_prior_per_target_index = ious.max(0)

#     for target_index, prior_index in enumerate(best_prior_per_target_index):
#         best_target_per_prior_index[prior_index] = target_index
#     # 2.0 is used to make sure every target has a prior assigned
#     best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
#     # size: num_priors
#     labels = gt_labels[best_target_per_prior_index]
#     labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
#     boxes = gt_boxes[best_target_per_prior_index]
#     return boxes, labels


# def hard_negative_mining(loss, labels, neg_pos_ratio):
#     """
#     It used to suppress the presence of a large number of negative prediction.
#     It works on image level not batch level.
#     For any example/image, it keeps all the positive predictions and
#      cut the number of negative predictions to make sure the ratio
#      between the negative examples and positive examples is no more
#      the given ratio for an image.
#     Args:
#         loss (N, num_priors): the loss for each example.
#         labels (N, num_priors): the labels.
#         neg_pos_ratio:  the ratio between the negative examples and positive examples.
#     """
#     pos_mask = labels > 0
#     num_pos = pos_mask.long().sum(dim=1, keepdim=True)
#     num_neg = num_pos * neg_pos_ratio

#     loss[pos_mask] = -math.inf
#     _, indexes = loss.sort(dim=1, descending=True)
#     _, orders = indexes.sort(dim=1)
#     neg_mask = orders < num_neg
#     return pos_mask | neg_mask
