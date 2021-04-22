from typing import Tuple, List, Dict
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
        top_k: int = 10,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.2,
        nms=None
    ) -> None:
        """
        """
        self.config = None
        self.num_classes = num_classes
        self.background_label = 0
        self.top_k = top_k
        self.nms_threshold = 0.5
        self.score_threshold = 0.2

        self.nms = fast_nms
        if self.nms is None:
            self.nms = fast_nms

    def __call__(
        self,
        preds: Dict[str, Tensor],
        image_sizes: List[Tuple[int]]
    ) -> List[Dict[str, Tensor]]:
        """
        """
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_masks = preds['mask_coefs']
        prior_boxes = preds['prior_boxes']
        proto_masks = preds['proto_masks']

        batch_size = pred_boxes.size(0)
        num_prior_boxes = prior_boxes.size(0)
        pred_scores = preds['scores'].view(
            batch_size, num_prior_boxes, self.num_classes).transpose(2, 1).contiguous()

        # test_scores, test_index = torch.max(preds['scores'], dim=1)

        return_list = []
        for i, image_size in enumerate(image_sizes):
            decoded_boxes = decode(pred_boxes[i], prior_boxes)
            results = self._filter_overlaps(i, decoded_boxes, pred_masks, pred_scores)
            results['proto_masks'] = proto_masks[i]

            return_list.append(_convert_boxes_and_masks(results, image_size))
            # return_list.append(results)

        for result in return_list:
            scores = result['scores'].detach().cpu()
            sorted_index = range(len(scores))[:self.top_k]
            # sorted_index = scores.argsort(0, descending=True)[:5]

            boxes = result['boxes'][sorted_index]
            labels = result['labels'][sorted_index]
            scores = scores[sorted_index]
            masks = result['masks'][sorted_index]

            result['boxes'] = boxes
            result['scores'] = scores
            result['labels'] = labels
            result['masks'] = masks

        return return_list

    def _filter_overlaps(
        self,
        batch_index,
        decoded_boxes,
        pred_masks,
        pred_scores,
    ) -> Dict[str, Tensor]:
        scores = pred_scores[batch_index, 1:, :]
        max_scores, _ = torch.max(scores, dim=0)

        keep = (max_scores > 0.05)
        scores = scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = pred_masks[batch_index, keep, :]

        if scores.size(1) == 0:
            return None

        # print(boxes.size(), scores.size())
        # boxes, masks, labels, scores = hard_nms(boxes, masks, scores)
        boxes, masks, labels, scores = self.nms(boxes, masks, scores)
        # print('fast_nms', boxes.size())

        return_dict = {
            'boxes': boxes,
            'mask_coefs': masks,
            'scores': scores,
            'labels': labels,
        }

        return return_dict


def _convert_boxes_and_masks(preds, size):
    """
    Args:
        preds
        size (): (h, w)

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
    for _class in range(num_classes):
        class_scores = scores[_class, :]
        conf_mask = class_scores > conf_thresh
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
    print('nms', boxes.size())

    # Undo the multiplication above
    return boxes[idx] / 550, masks[idx], classes, scores