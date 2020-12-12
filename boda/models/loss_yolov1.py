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
    def __init__(self, config, lambda_coord=None, lambda_noobj=None):
        super().__init__(config)
        self.config = config

        self.lambda_noobj = lambda_noobj
        if self.lambda_noobj is None:
            self.lambda_noobj = config.lambda_noobj

        self.lambda_coord = lambda_coord
        if self.lambda_coord is None:
            self.lambda_coord = config.lambda_coord

    def encode(self, targets):
        bs = len(targets)

        gs = self.config.num_grids
        nb = self.config.num_boxes
        nc = self.config.num_classes

        cell_size = 1.0 / gs
        h, w = self.config.max_size

        outputs = torch.zeros(bs, gs, gs, 5*nb+nc, device='cuda')
        for batch, target in enumerate(targets):
            boxes = target['boxes']
            boxes /= torch.tensor([[w, h, w, h]], dtype=torch.float, device=self.config.device).expand_as(boxes)
            boxes = xyxy_to_cxywh(boxes)
            for box_id, box in enumerate(boxes):
                ij = (box[:2] / cell_size).ceil() - 1.0
                i, j = int(ij[0]), int(ij[1])
                x0y0 = ij * cell_size
                box[:2] = (box[:2] - x0y0) / cell_size
                for k in range(0, 5*nb, 5):
                    outputs[batch, j, i, k:k+4] = box
                    outputs[batch, j, i, k+4] = 1.0
            
                labels = target['labels'][box_id].view(-1, 1)
                labels = torch.zeros(labels.size(0), nc, device='cuda').scatter(1, labels, 1)
                outputs[batch, j, i, 5*nb:] = labels.squeeze(0)

        outputs = outputs.view(bs, -1, 5*self.config.num_boxes+self.config.num_classes)
        boxes = outputs[..., :5*self.config.num_boxes].contiguous().view(bs, -1, 5)
        scores = boxes[..., 4]
        boxes = boxes[..., :4]
        labels = outputs[..., 5*self.config.num_boxes:]
        labels = labels.repeat(1, 2, 1)

        preds = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels}

        return preds

    def forward(self, inputs, targets):
        """
            inputs (Dict[str, Tensor])
            targets (Dict[str, Tensor])
        """
        self.check_targets(targets)
        targets = self.copy_targets(targets)
        targets = self.encode(targets)

        bs = self.config.batch_size
        gs = self.config.num_grids
        nb = self.config.num_boxes
        nc = self.config.num_classes

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

        loss_noobj = F.mse_loss(noobj_pred_scores, noobj_true_scores, reduction='sum')

        coord_response_mask = torch.zeros_like(true_scores, dtype=torch.bool, device=self.config.device)
        coord_not_response_mask = torch.ones_like(true_scores, dtype=torch.bool, device=self.config.device)

        iou_targets = torch.zeros_like(true_scores, device=self.config.device)

        _pred_boxes = torch.zeros_like(pred_boxes, device=self.config.device)
        _pred_boxes[:, :2] = pred_boxes[:, :2]/gs - 0.5*pred_boxes[:, 2:]
        _pred_boxes[:, 2:] = pred_boxes[:, :2]/gs + 0.5*pred_boxes[:, 2:]

        _true_boxes = torch.zeros_like(true_boxes, device=self.config.device)
        _true_boxes[:, :2] = true_boxes[:, :2]/gs - 0.5*true_boxes[:, 2:]
        _true_boxes[:, 2:] = true_boxes[:, :2]/gs + 0.5*true_boxes[:, 2:]

        iou = jaccard(_pred_boxes, _true_boxes)
        max_iou, max_index = iou.max(0)
        # print(max_iou)
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

        loss_boxes = (self.lambda_coord * (loss_xy + loss_wh)) / bs

        loss_class = F.mse_loss(pred_labels, true_labels, reduction='sum') / bs

        # loss_wh = torch.sum(torch.pow(torch.sqrt(response_boxes_targets[:, 3]) - torch.sqrt(response_boxes_preds[:, 3]), 2) \
        # + torch.pow(torch.sqrt(response_boxes_targets[:, 4]) - torch.sqrt(response_boxes_preds[:, 4]), 2))
        
        # loss_boxes = (self.lambda_coord * (loss_xy + loss_wh)) / bs 
        # # loss_boxes = (self.lambda_coord * (loss_xy)) / bs 

        loss_obj = F.mse_loss(
            pred_response_scores, iou_targets, reduction='sum')

        loss_scores = (loss_obj + (self.lambda_noobj * loss_noobj)) / bs
        # # Class probability loss for the cells which contain objects.
        # loss_class = F.mse_loss(class_preds, class_targets, reduction='sum') / bs

        losses = {
            'loss_boxes': loss_boxes,
            'loss_scores': loss_scores,
            'loss_class': loss_class,
        }

        return losses
