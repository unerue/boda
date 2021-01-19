import sys
import copy
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

    def encode(self, targets, num_grids, num_boxes, num_classes, device):
        """
        Args:
            Pascal VOC 
            targets ():
                boxes (float): [min x, min y, max x max y] [300, 400, 500, 700]
                labels (int): 10
        
        Returns:
            boxes [0.0 ~ 1.0, , ,  ~]
            labels [1]

        Returns:
            7 x 7 x 30
        """
        batch_size = len(targets)

        cell_size = 1.0 / num_grids
        h, w = self.max_size

        outputs = torch.zeros(batch_size, num_grids, num_grids, 5*num_boxes+num_classes, device=device)
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
                    torch.zeros(labels.size(0), num_classes, device=device).scatter(1, labels, 1)
                outputs[batch, j, i, 5*num_boxes:] = labels.squeeze(0)

        # outputs = outputs.view(batch_size, -1, 5*num_boxes+num_classes)
        # boxes = outputs[..., :5*num_boxes].contiguous().view(batch_size, -1, 5)
        # scores = boxes[..., 4]
        # boxes = boxes[..., :4]

        # labels = outputs[..., 5*num_boxes:]
        # # labels = labels.view(-1, num_classes)
        # # repeat = torch.LongTensor([num_boxes]).repeat(labels.size(0)).to(device)
        # # labels = torch.repeat_interleave(labels, repeat, dim=0)
        # # labels = labels.view(batch_size, 98, num_classes)

        # return_dict = {
        #     'boxes': boxes,
        #     'scores': scores,
        #     'labels': labels}

        # return return_dict
        return outputs

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
        targets = self.encode(targets, self.num_grids, self.num_boxes, 20, 'cuda')
        # print(inputs.size())
        # print(targets.size())
        batch_size = inputs.size(0)
        device = inputs.device

        coord_mask = targets[..., 4] > 0
        noobj_mask = targets[..., 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(targets)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(targets)

        pred_noobj = inputs[noobj_mask].view(-1, 30)
        true_noobj = targets[noobj_mask].view(-1, 30)

        inputs = inputs[coord_mask].view(-1, 30)
        pred_boxes = inputs[:, :5*2].contiguous().view(-1, 5)
        pred_labels = inputs[:, 5*2:]

        targets = targets[coord_mask].view(-1, 30)
        true_boxes = targets[:, :5*2].contiguous().view(-1, 5)
        true_labels = targets[:, 5*2:]
        # print(preds.size(), trues.size())
        # print(pred_boxes.size(), true_boxes.size())
        # print(pred_labels.size(), true_labels.size())
        
        # pred_noobj = inputs[noobj_mask].view(-1, 30)
        # true_noobj = targets[noobj_mask].view(-1, 30)

        # sys.exit()
        # batch_size = inputs['boxes'].size(0)
        # device = inputs['boxes'].device
        # num_classes = inputs['labels'].size(2)

        # self.check_targets(targets)
        # # targets = copy.deepcopy(self.copy_targets(targets))
        # targets = self.copy_targets(targets)

        # targets = self.encode(targets, self.num_grids, self.num_boxes, num_classes, device)

        # coord_mask = targets['scores'] > 0
        # noobj_mask = targets['scores'] == 0
        # # print(coord_mask[:, :49].size())
        # # dummy = torch.zeros(batch_size, 7, 7, 30)
        # # coord_mask = coord_mask.unsqueeze(-1).expand_as(dummy)
        # pred_boxes = inputs['boxes'][coord_mask]
        # pred_scores = inputs['scores'][coord_mask]
        # pred_labels = inputs['labels'][coord_mask[:, :self.num_grids*self.num_grids]]
        # # print(pred_boxes[50], pred_boxes.dtype)
        # # print(pred_scores[50], pred_scores.dtype)
        # # print(pred_labels[50], pred_labels.dtype)
        # true_boxes = targets['boxes'][coord_mask]
        # true_scores = targets['scores'][coord_mask]
        # true_labels = targets['labels'][coord_mask[:, :self.num_grids*self.num_grids]]

        # pred_noobj_scores = inputs['scores'][noobj_mask]
        # true_noobj_scores = targets['scores'][noobj_mask]

        noobj_conf_mask = torch.zeros_like(pred_noobj, dtype=torch.bool)
        for b in range(self.num_boxes):
            noobj_conf_mask[:, 4+b*5] = 1 # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1

        pred_noobj_scores = pred_noobj[noobj_conf_mask]       # [n_noobj, 2=len([conf1, conf2])]
        true_noobj_scores = true_noobj[noobj_conf_mask]   # [n_noobj, 2=len([conf1, conf2])]
        # print(pred_noobj_scores[0])
        # print(true_noobj_scores[0])
        loss_noobj = \
            F.mse_loss(pred_noobj_scores, true_noobj_scores, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.zeros_like(pred_boxes, dtype=torch.bool)
        # coord_not_response_mask = torch.ones_like(true_boxes, dtype=torch.bool)
        coord_not_response_mask = torch.zeros_like(true_boxes, dtype=torch.bool)
        matched_boxes = torch.zeros_like(true_boxes)

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, true_boxes.size(0), 2):
            boxes = pred_boxes[i:i+2] # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            _pred_boxes = torch.zeros_like(boxes, dtype=torch.float32) # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            _pred_boxes[:, :2] = boxes[:, :2]/self.num_grids - 0.5 * boxes[:, 2:4]
            _pred_boxes[:, 2:4] = boxes[:, :2]/self.num_grids + 0.5 * boxes[:, 2:4]

            boxes = true_boxes[i].view(-1, 5)  # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            # target = true_boxes[i].view(-1, 5)  # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            _true_boxes = torch.zeros_like(boxes, dtype=torch.float32)  # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            _true_boxes[:, :2] = boxes[:, :2]/self.num_grids - 0.5 * boxes[:, 2:4]
            _true_boxes[:, 2:4] = boxes[:, :2]/self.num_grids + 0.5 * boxes[:, 2:4]
            # target_xyxy[:, :2] = target[:, :2]
            # target_xyxy[:, 2:4] = target[:, 2:4]

            iou = jaccard(_pred_boxes[:, :4], _true_boxes[:, :4])  # [B, 1]
            max_iou, max_index = iou.max(0)
            # max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            # coord_not_response_mask[i+max_index] = 0
            coord_not_response_mask[i+1-max_index] = 1

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            # print(max_iou.size())
            matched_boxes[i+max_index, 4] = max_iou
            # bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = max_iou

        # bbox_target_iou = bbox_target_iou.cuda()
        # print(coord_response_mask[0])
        # print(coord_not_response_mask[0])

        # BBox location/size and objectness loss for the response bboxes.
        pred_response_boxes = pred_boxes[coord_response_mask].view(-1, 5)      # [n_response, 5]
        true_response_boxes = true_boxes[coord_response_mask].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        matched_boxes = matched_boxes[coord_response_mask].view(-1, 5)     # [n_response, 5], only the last 1=(conf,) is used
        # print(bbox_pred_response[0], bbox_target_response[0])
        losses = {}
        loss_xy = F.mse_loss(
            pred_response_boxes[:, :2], true_response_boxes[:, :2], reduction='sum')
        loss_wh = F.mse_loss(
            torch.sqrt(pred_response_boxes[:, 2:4]),
            torch.sqrt(true_response_boxes[:, 2:4]), reduction='sum')

        losses['B'] = (self.lambda_coord * (loss_xy + loss_wh)) / batch_size

        # print(pred_response_boxes[0])
        # print(matched_boxes[0])
        loss_obj = F.mse_loss(pred_response_boxes[:, 4], matched_boxes[:, 4], reduction='sum')

        pred_not_response_boxes = pred_boxes[coord_not_response_mask].view(-1, 5)
        true_not_response_boxes = true_boxes[coord_not_response_mask].view(-1, 5)
        true_not_response_boxes[:, 4] = 0
        # # print(pred_not_response_boxes[0])
        # # print(true_not_response_boxes[0])

        loss_not = F.mse_loss(
            pred_not_response_boxes[:, 4],
            true_not_response_boxes[:, 4], reduction='sum')

        # losses['S'] = (loss_obj + self.lambda_noobj * loss_noobj) / batch_size
        losses['S'] = (loss_not + loss_obj + self.lambda_noobj * loss_noobj) / batch_size

        # Class probability loss for the cells which contain objects.
        # print(pred_labels.size(), true_labels.size())
        losses['C'] = F.mse_loss(pred_labels, true_labels, reduction='sum') / batch_size

        # Total loss
        # loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        # loss = loss / float(batch_size)

        return losses

        # # self.check_targets(targets)
        # # print('inputs:', inputs[0]['boxes'][0])
        # # print('targets:', targets[0]['boxes'][0])
        # # inputs = copy.deepcopy(self.copy_targets(inputs))
        # # targets = copy.deepcopy(self.copy_targets(targets))
        # # print(inputs[0]['boxes'][0], id(inputs[0]['boxes']))
        # # print(targets[0]['boxes'][0], id(targets[0]['boxes']))
        # # batch_size = inputs['boxes'].size(0)
        # # num_classes = inputs['labels'].size(2)
        # # batch_size = 64
        # # num_classes = 20
        # # # print()
        # # inputs = self.encode(inputs, self.num_grids, self.num_boxes, num_classes)
        # # # print(inputs['boxes'][0][0], inputs['boxes'].dtype)
        # # # print('\n\n')
        # # targets = self.encode(targets, self.num_grids, self.num_boxes, num_classes)
        # # print(targets['boxes'][0][0], targets['boxes'].dtype)
        # # print()
        # # coord_mask = targets['scores'] > 0
        # # noobj_mask = targets['scores'] == 0

        # # pred_boxes = inputs['boxes'][coord_mask]
        # # pred_scores = inputs['scores'][coord_mask]
        # # pred_labels = inputs['labels'][coord_mask]

        # # true_boxes = targets['boxes'][coord_mask]
        # # true_scores = targets['scores'][coord_mask]
        # # true_labels = targets['labels'][coord_mask]

        # # pred_noobj_scores = inputs['scores'][noobj_mask]
        # # true_noobj_scores = targets['scores'][noobj_mask]
        # #####################
        # # print('pred', inputs['boxes'][0][0], inputs['boxes'].dtype)
        # # print('true', targets['boxes'][0][0], targets['boxes'].dtype)
        # # print(torch.sum(noobj_pred_scores))
        # # print(torch.sum(noobj_true_scores))
        # loss_noobj = \
        #     F.mse_loss(pred_noobj_scores, true_noobj_scores, reduction='sum')
        # # print('loss_noobj', loss_noobj)
        # coord_response_mask = torch.zeros_like(true_scores, dtype=torch.bool)
        # coord_not_response_mask = torch.ones_like(true_scores, dtype=torch.bool)

        # # iou_targets = torch.zeros_like(true_scores, device=true_scores.device)
        # # Rescale (cx, cy, w, h)
        # _pred_boxes = torch.zeros_like(pred_boxes, dtype=torch.float32)
        # _pred_boxes[:, :2] = pred_boxes[:, :2]/self.num_grids - 0.5*pred_boxes[:, 2:]
        # _pred_boxes[:, 2:] = pred_boxes[:, :2]/self.num_grids + 0.5*pred_boxes[:, 2:]

        # # _true_boxes = torch.zeros_like(true_boxes, dtype=torch.float32)
        # # _true_boxes[:, :2] = true_boxes[:, :2]/self.num_grids - 0.5*true_boxes[:, 2:]
        # # _true_boxes[:, 2:] = true_boxes[:, :2]/self.num_grids + 0.5*true_boxes[:, 2:]
        # # print(pred_boxes[50], _pred_boxes.dtype)
        # # print(_true_boxes[50], _pred_boxes.dtype)
        # # _pred_boxes = _pred_boxes.half()
        # # _true_boxes = _true_boxes.half()
        # iou = jaccard(_pred_boxes, true_boxes)
        # # print('iou')
        # # print(iou)
        # # print()
        # max_iou, max_index = iou.max(0)
        # # print(max_iou, max_iou.size())
        # # print(max_index, max_index.size())

        # coord_response_mask[max_index] = 1
        # coord_not_response_mask[max_index] = 0

        # pred_response_boxes = pred_boxes[coord_response_mask]
        # pred_response_scores = pred_scores[coord_response_mask]

        # true_response_boxes = true_boxes[coord_response_mask]
        # iou_response_scores = max_iou[coord_response_mask]
        # # print(pred_response_boxes[0], true_response_boxes[0])
        # loss_xy = F.mse_loss(
        #     pred_response_boxes[:, :2],
        #     true_response_boxes[:, :2], reduction='sum')

        # loss_wh = F.mse_loss(
        #     torch.sqrt(pred_response_boxes[:, 2:]),
        #     torch.sqrt(true_response_boxes[:, 2:]), reduction='sum')

        # losses = {}
        # losses['B'] = (self.lambda_coord * (loss_xy + loss_wh)) / batch_size

        # loss_obj = F.mse_loss(
        #     pred_response_scores, iou_response_scores, reduction='sum')
        # losses['S'] = (loss_obj + (self.lambda_noobj * loss_noobj)) / batch_size

        # losses['C'] = F.mse_loss(pred_labels, true_labels, reduction='sum') / batch_size

        # return losses
