import sys
from typing import Tuple, List, Dict, Any, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import LossFunction
from ..utils.bbox import xyxy_to_cxywh
from ..utils.bbox import jaccard


class Yolov1Loss(LossFunction):
    """Loss Function for YOLOv1

    Arguments:
        lambda_coord ():
        lambda_noobj ():

    """
    def __init__(
        self,
        config=None,
        lambda_coord=5.0,
        lambda_noobj=0.5,
        **kwargs
    ) -> None:
        super().__init__()
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.class_scale = 1
        self.coord_scale = 5
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.num_boxes = kwargs.get('num_boxes', 2)
        self.num_grids = kwargs.get('num_grids', 7)
        self.max_size = kwargs.get('max_size', (448, 448))

        if config is not None:
            for k, v in config.to_dict().items():
                setattr(self, k, v)

    def encode(self, targets, num_grids=7, num_boxes=2, num_classes=20):
        """
        Args:

        """
        batch_size = len(targets)

        cell_size = 1.0 / num_grids
        h, w = self.max_size

        outputs = torch.zeros(batch_size, num_grids, num_grids, 5*num_boxes+num_classes, device='cuda')
        for batch, target in enumerate(targets):
            boxes = target['boxes']
            boxes /= torch.tensor(
                [[w, h, w, h]], dtype=torch.float,
                device=boxes.device).expand_as(boxes)

            boxes = xyxy_to_cxywh(boxes)
            for box_id, box in enumerate(boxes):
                ij = (box[:2] / cell_size).ceil() - 1.0
                i, j = int(ij[0]), int(ij[1])
                x0y0 = ij * cell_size
                box[:2] = (box[:2] - x0y0) / cell_size
                for k in range(0, 5*num_boxes, 5):
                    outputs[batch, j, i, k:k+4] = box
                    outputs[batch, j, i, k+4] = 1.0

                labels = target['labels'][box_id].view(-1, 1)
                labels = \
                    torch.zeros(labels.size(0), num_classes, device=boxes.device).scatter(1, labels, 1)
                outputs[batch, j, i, 5*num_boxes:] = labels.squeeze(0)

        outputs = outputs.view(batch_size, -1, 5*num_boxes+num_classes)
        boxes = outputs[..., :5*num_boxes].contiguous().view(batch_size, -1, 5)
        scores = boxes[..., 4]
        boxes = boxes[..., :4]
        labels = outputs[..., 5*num_boxes:]
        labels = labels.repeat(1, 2, 1)

        return_dict = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels}

        return return_dict

    def forward(
        self,
        inputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """
            inputs (Dict[str, Tensor])
            targets (Dict[str, Tensor])
        """
        self.check_targets(targets)
        targets = self.copy_targets(targets)
        batch_size = inputs['boxes'].size(0)
        num_classes = inputs['labels'].size(2)

        targets = self.encode(targets, self.num_grids, self.num_boxes, num_classes)

        coord_mask = targets['scores'] > 0
        noobj_mask = targets['scores'] == 0

        pred_boxes = inputs['boxes'][coord_mask]
        pred_scores = inputs['scores'][coord_mask]
        pred_labels = inputs['labels'][coord_mask]

        true_boxes = targets['boxes'][coord_mask]
        true_scores = targets['scores'][coord_mask]
        true_labels = targets['labels'][coord_mask]

        noobj_pred_scores = inputs['scores'][noobj_mask]
        noobj_true_scores = targets['scores'][noobj_mask]

        loss_noobj = \
            F.mse_loss(
                noobj_pred_scores, noobj_true_scores, reduction='sum')

        coord_response_mask = \
            torch.zeros_like(true_scores, dtype=torch.bool, device=true_scores.device)
        coord_not_response_mask = \
            torch.ones_like(true_scores, dtype=torch.bool, device=true_scores.device)

        iou_targets = torch.zeros_like(true_scores, device=true_scores.device)

        _pred_boxes = torch.zeros_like(pred_boxes, device=pred_boxes.device)
        _pred_boxes[:, :2] = pred_boxes[:, :2]/self.num_grids - 0.5*pred_boxes[:, 2:]
        _pred_boxes[:, 2:] = pred_boxes[:, :2]/self.num_grids + 0.5*pred_boxes[:, 2:]

        _true_boxes = torch.zeros_like(true_boxes, device=true_boxes.device)
        _true_boxes[:, :2] = true_boxes[:, :2]/self.num_grids - 0.5*true_boxes[:, 2:]
        _true_boxes[:, 2:] = true_boxes[:, :2]/self.num_grids + 0.5*true_boxes[:, 2:]

        iou = jaccard(_pred_boxes, _true_boxes)
        max_iou, max_index = iou.max(0)

        coord_response_mask[max_index] = 1
        coord_not_response_mask[max_index] = 0

        response_boxes_preds = pred_boxes[coord_response_mask]
        pred_response_scores = pred_scores[coord_response_mask]
        response_boxes_targets = true_boxes[coord_response_mask]
        iou_targets = max_iou[coord_response_mask]

        loss_xy = F.mse_loss(
            response_boxes_preds[:, :2],
            response_boxes_targets[:, :2], reduction='sum')

        loss_wh = F.mse_loss(
            torch.sqrt(response_boxes_preds[:, 2:4]),
            torch.sqrt(response_boxes_targets[:, 2:4]), reduction='sum')

        losses = {}
        losses['B'] = (self.lambda_coord * (loss_xy + loss_wh)) / batch_size

        loss_obj = F.mse_loss(
            pred_response_scores, iou_targets, reduction='sum')
        losses['S'] = (loss_obj + (self.lambda_noobj * loss_noobj)) / batch_size

        losses['C'] = F.mse_loss(pred_labels, true_labels, reduction='sum') / batch_size

        return losses
