import torch
from torch import nn, Tensor


def point_form(boxes: Tensor) -> Tensor:
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Argument:
        boxes (Tensor): [[cx, cy, w, h]] center-size default boxes from priorbox layers.
    Return:
        boxes: (Tensor) Converted [[xmin, ymin, xmax, ymax]] form of boxes.
    """
    return torch.cat(
        (boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), 1)


def xyxy_to_cxywh(boxes: Tensor):
    """Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Argument:
        boxes (Tensor): [xmin, ymin, xmax, ymax] point_form boxes
    Return:
        boxes (Tensor): Converted [cx, cy, w, h] form of boxes.
    """
    return torch.cat(
        ((boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2]), dim=1)


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




