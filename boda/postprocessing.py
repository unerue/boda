from collections import defaultdict
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms


def decode(boxes: Tensor, prior_boxes: Tensor, variances: List[float] = [0.1, 0.2]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.

    https://github.com/Hakuyume/chainer-ssd

    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors, 4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: (`List[float]`) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat(
        (
            prior_boxes[:, :2] + boxes[:, :2] * variances[0] * prior_boxes[:, 2:],
            prior_boxes[:, 2:] * torch.exp(boxes[:, 2:] * variances[1]),
        ),
        dim=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def sanitize_coordinates(
    _x1, _x2, img_size: int, padding: int = 0, cast: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.
    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


def crop(masks, boxes, padding: int = 1) -> Tensor:
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).
    Args:
        # TODO: torchvision mask rcnn masks UInt8Tensor[N, H, W]
        # TODO: torchvision boxes FloatTensor[N, 4]
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = (
        torch.arange(w, device=masks.device, dtype=x1.dtype)
        .view(1, -1, 1)
        .expand(h, w, n)
    )
    cols = (
        torch.arange(h, device=masks.device, dtype=x1.dtype)
        .view(-1, 1, 1)
        .expand(h, w, n)
    )

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


class PostprocessOutputs:
    def __init__(
        self,
        num_classes: int = 80,
        top_k: int = 10,
        nms_threshold: float = 0.3,
        score_threshold: float = 0.2,
    ) -> None:
        """
        Args:
            num_classes (int)
            top_k
            nms_threshold
            score_threshold
            nms ()
        """
        self.config = None
        self.num_classes = num_classes + 1
        self.background_label = 0
        self.top_k = top_k
        self.nms_threshold = 0.5
        self.score_threshold = 0.2

        self.nms = batched_nms
        # if self.nms is None:
        #     self.nms = fast_nms

    def __call__(
        self, preds: Dict[str, Tensor], image_sizes: List[Tuple[int]]
    ) -> List[Dict[str, Tensor]]:
        """ """
        pred_boxes = None
        pred_scores = None
        default_boxes = None
        pred_masks = None
        proto_masks = None
        if "boxes" in preds:
            pred_boxes = preds["boxes"]
        if "scores" in preds:
            pred_scores = preds["scores"]
        if "default_boxes" in preds:
            default_boxes = preds["default_boxes"]
        if "mask_coefs" in preds:
            pred_masks = preds["mask_coefs"]
        if "proto_masks" in preds:
            proto_masks = preds["proto_masks"]

        batch_size = pred_boxes.size(0)
        num_prior_boxes = default_boxes.size(0)
        pred_scores = (
            preds["scores"]
            .view(batch_size, num_prior_boxes, self.num_classes)
            .transpose(2, 1)
            .contiguous()
        )

        # test_scores, test_index = torch.max(preds['scores'], dim=1)

        return_list = []
        # print(image_sizes)
        for i, image_size in enumerate(image_sizes):
            print(i, proto_masks.size())
            decoded_boxes = decode(pred_boxes[i], default_boxes)
            results = self._filter_overlaps(i, decoded_boxes, pred_masks, pred_scores)
            print(proto_masks[i].dtype)
            results["proto_masks"] = proto_masks[i]

            return_list.append(_convert_boxes_and_masks(results, image_size))
            # return_list.append(results)

        for result in return_list:
            scores = result["scores"].detach().cpu()
            sorted_index = range(len(scores))[: self.top_k]
            # sorted_index = scores.argsort(0, descending=True)[:5]

            boxes = result["boxes"][sorted_index]
            labels = result["labels"][sorted_index]
            scores = scores[sorted_index]
            masks = result["masks"][sorted_index]

            result["boxes"] = boxes
            result["scores"] = scores
            result["labels"] = labels
            result["masks"] = masks

        return return_list

    def _filter_overlaps(
        self,
        batch_index,
        decoded_boxes,
        pred_masks,
        pred_scores,
    ) -> Dict[str, Tensor]:
        scores = pred_scores[batch_index, 1:, :]
        max_scores, labels = torch.max(scores, dim=0)

        keep = max_scores > self.score_threshold  # 0.05
        scores = scores[:, keep]
        boxes = decoded_boxes[keep, :]
        labels = labels[keep]
        masks = pred_masks[batch_index, keep, :]

        if scores.size(1) == 0:
            return None

        # print(max_scores[0], max_class[0])
        print(boxes.size(), scores.size(), keep.size(), labels.size())
        # boxes, masks, labels, scores = self.nms(boxes, scores, keep, iou_threshold=0.3)

        return_dict = defaultdict()
        for _class in range(scores.size(0)):
            _scores = scores[_class, :]
            indices = self.nms(boxes, _scores, labels, iou_threshold=0.3)

        return_dict["boxes"] = boxes[indices]
        return_dict["scores"] = _scores[indices]
        return_dict["mask_coefs"] = masks[indices]
        return_dict["labels"] = labels[indices]

        return dict(return_dict)


def _convert_boxes_and_masks(preds, size):
    """
    Args:
        preds
        size (): (h, w)

    """
    h, w = size
    boxes = preds["boxes"]
    mask_coefs = preds["mask_coefs"]
    proto_masks = preds["proto_masks"]

    masks = proto_masks @ mask_coefs.t()
    masks = torch.sigmoid(masks)

    masks = crop(masks, boxes)
    masks = F.interpolate(
        masks.unsqueeze(0), (h, w), mode="bilinear", align_corners=False
    ).squeeze(0)
    masks.gt_(0.5)  # Binarize the masks

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
        boxes[:, 0], boxes[:, 2], w, cast=False
    )
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(
        boxes[:, 1], boxes[:, 3], h, cast=False
    )
    boxes = boxes.long()

    preds["boxes"] = boxes
    preds["masks"] = masks

    del preds["proto_masks"]
    del preds["mask_coefs"]

    return preds
