from typing import Tuple
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from ...ops.box import decode, sanitize_coordinates, crop, jaccard
from ...ops.nms import fast_nms, torchvision_nms


class YolactInference:
    def __init__(
        self,
        num_classes: int = 81,
    ) -> None:
        self.config = None
        self.num_classes = num_classes
        self.background_label = 0
        self.top_k = 5
        self.nms_threshold = 0.5
        self.score_threshold = 0.2

        self.use_cross_class_nms = False
        self.use_fast_nms = False

        self.nms = fast_nms

    def __call__(self, preds):
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_masks = preds['mask_coefs']
        prior_boxes = preds['prior_boxes']
        proto_masks = preds['proto_masks']

        batch_size = pred_boxes.size(0)
        num_prior_boxes = prior_boxes.size(0)
        print(preds['scores'].size())
        pred_scores = preds['scores'].view(
            batch_size, num_prior_boxes, self.num_classes).transpose(2, 1).contiguous()

        test_scores, test_index = torch.max(preds['scores'], dim=1)
        print(test_scores.size(), test_index.size())

        outputs = []
        for i in range(batch_size):
            decoded_boxes = decode(pred_boxes[i], prior_boxes)
            results = self.detect(i, decoded_boxes, pred_masks, pred_scores)
            results['proto_masks'] = proto_masks[i]
            outputs.append(results)

        return outputs

    def detect(
        self,
        batch_index,
        decoded_boxes,
        pred_masks,
        pred_scores,
    ):
        scores = pred_scores[batch_index, 1:, :]
        max_scores, _ = torch.max(scores, dim=0)

        keep = (max_scores > 0.05)
        scores = scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = pred_masks[batch_index, keep, :]

        if scores.size(1) == 0:
            return None

        print(boxes.size(), scores.size())
        hard_nms(boxes, masks, scores)
        boxes, masks, labels, scores = self.nms(boxes, masks, scores)
        print('fast_nms', boxes.size())

        return_dict = {
            'boxes': boxes,
            'mask_coefs': masks,
            'scores': scores,
            'labels': labels,
        }

        return return_dict


def convert_masks_and_boxes(preds, size):
    """
    Args:
        preds
        size (): (w, h)

    """
    h, w = size
    boxes = preds['boxes']
    mask_coefs = preds['mask_coefs']
    proto_masks = preds['proto_masks']

    masks = proto_masks @ mask_coefs.t()
    masks = torch.sigmoid(masks)

    masks = crop(masks, boxes)
    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)
    masks.gt_(0.5)  # Binarize the masks

    boxes[:, 0], boxes[:, 2] = \
        sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = \
        sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    preds['boxes'] = boxes
    preds['masks'] = masks

    del preds['proto_masks']
    del preds['mask_coefs']

    return preds


def hard_nms(boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
    from torchvision.ops import nms as torchvision_nms
    num_classes = scores.size(0)

    idx_lst = []
    cls_lst = []
    scr_lst = []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    boxes = boxes * 550
    # print(scores.size())
    for _cls in range(num_classes):
        cls_scores = scores[_cls, :]
        conf_mask = cls_scores > conf_thresh
        idx = torch.arange(cls_scores.size(0), device=boxes.device)

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue

        keep = torchvision_nms(boxes[conf_mask].cpu(), cls_scores.cpu(), iou_threshold)
        # keep = torch.Tensor(keep, device=boxes.device).long()

        idx_lst.append(idx[keep])
        cls_lst.append(keep * 0 + _cls)
        scr_lst.append(cls_scores[keep])
    
    idx     = torch.cat(idx_lst, dim=0)
    classes = torch.cat(cls_lst, dim=0)
    scores  = torch.cat(scr_lst, dim=0)

    scores, idx2 = scores.sort(0, descending=True)
    idx2 = idx2[:200]
    scores = scores[:200]

    idx = idx[idx2]
    classes = classes[idx2]
    print('nms', boxes.size())

    # Undo the multiplication above
    return boxes[idx] / 550, masks[idx], classes, scores