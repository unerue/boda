import torch
from torch import Tensor
from torchvision.ops import nms as torchvision_nms
from .box import jaccard


def hard_nms(
    boxes,
    scores,
    masks=None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05
) -> Tensor:
    """
    Args:
        boxes (Tensor): Tensor[N, 4]
        scores (Tensor): Tensor[]
        masks (Tensor): default is None
        iou_threshold (float)
        score_threshold (float)
    """
    num_classes = scores.size(0)

    idx_lst = []
    cls_lst = []
    scr_lst = []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    # boxes = boxes * 550
    for _class in range(num_classes):
        class_scores = scores[_class, :]
        conf_mask = class_scores > score_threshold
        idx = torch.arange(class_scores.size(0), device=boxes.device)

        cls_scores = class_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue

        keep = torchvision_nms(boxes[conf_mask].cpu(), cls_scores.cpu(), iou_threshold)
        # keep = torch.Tensor(keep, device=boxes.device).long()

        idx_lst.append(idx[keep])
        cls_lst.append(keep * 0 + _class)
        scr_lst.append(cls_scores[keep])

    idx = torch.cat(idx_lst, dim=0)
    classes = torch.cat(cls_lst, dim=0)
    scores = torch.cat(scr_lst, dim=0)

    scores, idx2 = scores.sort(0, descending=True)
    idx2 = idx2[:200]
    scores = scores[:200]

    idx = idx[idx2]
    classes = classes[idx2]

    # Undo the multiplication above
    return boxes[idx] / 550, masks[idx], classes, scores


def fast_nms(
    boxes,
    scores,
    masks=None,
    iou_threshold: float = 0.5,
    top_k: int = 200,
    second_threshold: bool = False
) -> None:
    """
    """
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]

    num_classes, num_dets = idx.size()

    boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    if masks is not None:
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= iou_threshold)

    # We should also only keep detections over the confidence threshold, but at the cost of
    # maxing out your detection count for every image, you can just not do that. Because we
    # have such a minimal amount of computation per detection (matrix mulitplication only),
    # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
    # However, when you implement this in your method, you should do this second threshold.
    if second_threshold:
        keep *= (scores > 0.2)  # self.conf_thresh 0.2

    # Assign each kept detection to its corresponding class
    classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    scores = scores[keep]
    if masks is not None:
        masks = masks[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:200]
    scores = scores[:200]  # TODO: max_num_detection

    classes = classes[idx]
    boxes = boxes[idx]
    if masks is not None:
        masks = masks[idx]
        return boxes, masks, classes, scores
    else:
        return boxes, classes, scores
