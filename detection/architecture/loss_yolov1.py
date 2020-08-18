import sys

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional

from .loss_base import _check_targets
from ..configuration import yolov1_config
from ..utils import jaccard
# from torch.jit.annotations import Tuple, List, Dict, Any, Optional


class Yolov1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_boxes = yolov1_config.num_boxes
        self.num_classes = yolov1_config.dataset.num_classes
        self.grid_size = yolov1_config.grid_size

        self.lambda_coord = yolov1_config.lambda_coord
        self.lambda_noobj = yolov1_config.lambda_noobj

    def forward(self, inputs: List[Tensor], targets: List[Dict[str, Tensor]]) -> Tensor:
        """
        S * S * (B * 5 + C) Tensor
        where S denotes grid size and for each grid cell predicts B bounding boxes, 
        confidence for those boxes, and C class probabilites.
        """
        if self.training and targets is None:
            raise ValueError
        
        self.device = inputs.device
        bs = inputs.size(0)  # batch size
        nb = self.num_boxes  # B
        gs = inputs.size(2)  # S

        targets = self._transform_targets(targets)
        # print(inputs.size(), targets.size())
        # print(inputs.device, targets.device)
        
        # torch.Size([batch_size, S, S, 5*B+C])
        coord_mask = targets[..., 4] > 0
        noobj_mask = targets[..., 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(targets)#.to(self.device)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(targets)#.to(self.device)

        # print(coord_mask[0][3])
        # print(noobj_mask[0][3])
        # print('='*100)
        # print(coord_mask.size(), noobj_mask.size())
        # print(coord_mask.dtype, noobj_mask.dtype)
        # print(coord_mask.device, noobj_mask.device)

        # Covert predicted outputs
        # torch.Size([, 30]), torch.Size([, 5]), torch.Size([, 20])
        coord_preds = inputs[coord_mask].view(-1, 5*nb+self.num_classes)
        # [n_coord x B, 5=len([x, y, w, h, conf])]
        boxes_preds = coord_preds[:, :5*nb].contiguous().view(-1, 5) 
        class_preds = coord_preds[:, 5*nb:]
        # print(coord_preds.size(), boxes_preds.size(), class_preds.size())
        # print(coord_preds.dtype, boxes_preds.dtype, class_preds.dtype)
        # print(coord_preds.device, boxes_preds.device, class_preds.device)
        # print(coord_preds)
        # print(boxes_preds)
        # print(class_preds)
        # print('='*100)
        # Covert targets
        coord_targets = targets[coord_mask].view(-1, 5*nb+self.num_classes)
        # [n_coord x B, 5=len([x, y, w, h, conf])]
        boxes_targets = coord_targets[:, :5*nb].contiguous().view(-1, 5)
        class_targets = coord_targets[:, 5*nb:]
        # print(coord_targets.size(), boxes_targets.size(), class_targets.size())
        # print(coord_targets.dtype, boxes_targets.dtype, class_targets.dtype)
        # print(coord_targets.device, boxes_targets.device, class_targets.device)
        # print(coord_targets)
        # print(boxes_targets)
        # print(class_targets)
        # print('='*100)
        # Compute loss for the cells with no object bbox.
        # pred tensor on the cells which do not contain objects. [n_noobj, N]
        # n_noobj: number of the cells which do not contain objects.
        noobj_preds = inputs[noobj_mask].view(-1, 5*nb+self.num_classes)
        # target tensor on the cells which do not contain objects. [n_noobj, N]
        # n_noobj: number of the cells which do not contain objects.
        noobj_targets = targets[noobj_mask].view(-1, 5*nb+self.num_classes)
        # print(noobj_preds.dtype, noobj_targets.dtype)
        # [n_noobj, N]
        noobj_conf_mask = torch.BoolTensor(noobj_preds.size()).fill_(0).to(self.device)
        # print(noobj_conf_mask[0])
        # print(noobj_preds.size(), noobj_targets.size(), noobj_conf_mask.size())
        # print(noobj_preds.dtype, noobj_targets.dtype, noobj_conf_mask.dtype)
        # print(noobj_preds.device, noobj_targets.device, noobj_conf_mask.device)
        # print('='*100)
        # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
        for b in range(nb):
            noobj_conf_mask[:, 4+b*5] = 1
        # print(noobj_conf_mask[noobj_conf_mask > 0])
        # [n_noobj, 2=len([conf1, conf2])]
        # [n_noobj, 2=len([conf1, conf2])]
        noobj_conf_preds = noobj_preds[noobj_conf_mask] 
        noobj_conf_targets = noobj_targets[noobj_conf_mask]
        # print(noobj_conf_preds.device, noobj_conf_targets.device)
        loss_noobj = F.mse_loss(noobj_conf_preds, noobj_conf_targets, reduction='sum')
        # print(loss_noobj)
        # Compute loss for the cells with objects.
        # We assign one predictor to be “responsible” for predicting an object based on which
        # prediction has the highest current IOU with the ground truth.
        # [n_coord x B, 5] # [n_coord x B, 5]
        coord_response_mask = torch.BoolTensor(boxes_targets.size()).fill_(False).to(self.device)
        coord_not_response_mask = torch.BoolTensor(boxes_targets.size()).fill_(True).to(self.device)
        # print(coord_response_mask.dtype, coord_not_response_mask.dtype)
        # print(coord_response_mask.device, coord_not_response_mask.device)
        # print(coord_response_mask.size(), coord_not_response_mask.size())
        
        # [n_coord x B, 5], only the last 1=(conf,) is used
        iou_targets = torch.zeros(boxes_targets.size(), device=self.device)
        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, boxes_targets.size(0), 2):
            # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            _preds = boxes_preds[i:i+nb]
            # [B, 5=len([x1, y1, x2, y2, conf])]
            xyxy_preds = torch.FloatTensor(_preds.size()).to(self.device)
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] 
            # are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            xyxy_preds[:, 0:2] = _preds[:, :2]/gs - 0.5 * _preds[:, 2:4]
            xyxy_preds[:, 2:4] = _preds[:, :2]/gs + 0.5 * _preds[:, 2:4]
            # target bbox at i-th cell. Because target boxes contained by each cell 
            # are identical in current implementation, enough to extract the first one.
            # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            _targets = boxes_targets[i].view(-1, 5)
            # [1, 5=len([x1, y1, x2, y2, conf])]
            xyxy_targets = torch.FloatTensor(_targets.size()).to(self.device)
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for 
            # cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            xyxy_targets[:, 0:2] = _targets[:, :2]/gs - 0.5 * _targets[:, 2:4]
            xyxy_targets[:, 2:4] = _targets[:, :2]/gs + 0.5 * _targets[:, 2:4]

            # iou = jaccard(xyxy_preds[..., :4], xyxy_targets[..., :4]) # [B, 1]
            iou = compute_iou(xyxy_preds[..., :4], xyxy_targets[..., :4]) # [B, 1]
            
            max_iou, max_index = iou.max(0)
            # print(max_iou, max_index)
            # print(max_iou.dtype, max_index.dtype)
            # print(max_iou.device, max_index.device)
            
            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0
            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            iou_targets[i+max_index, torch.LongTensor([4])] = max_iou
        
        # BBox location/size and objectness loss for the response bboxes.
        response_boxes_preds = boxes_preds[coord_response_mask].view(-1, 5)# .to('cuda')      # [n_response, 5]
        response_boxes_targets = boxes_targets[coord_response_mask].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        # print(response_boxes_preds.size(), response_boxes_targets.size())
        # print('-'*100)
        # print(response_boxes_preds)
        # print(response_boxes_targets)
        # print('-'*100)
        iou_targets = iou_targets[coord_response_mask].view(-1, 5)#.to('cuda')        # [n_response, 5], only the last 1=(conf,) is used
        # print(iou_targets)
        # print('-'*100)
        loss_xy = F.mse_loss(
            response_boxes_preds[:, :2], 
            response_boxes_targets[:, :2], reduction='sum')

        # loss_wh = F.mse_loss(
        #     torch.sqrt(torch.abs(response_boxes_preds[:, 2:4])), 
        #     torch.sqrt(response_boxes_targets[:, 2:4]), reduction='sum')
        loss_wh = F.mse_loss(
            torch.sqrt(response_boxes_preds[:, 2:4]), 
            torch.sqrt(response_boxes_targets[:, 2:4]), reduction='sum')
        # print(loss_xy.item(), loss_wh.item())
        loss_boxes = (self.lambda_coord * (loss_xy + loss_wh)) / bs

        loss_obj = F.mse_loss(
            response_boxes_preds[:, 4], iou_targets[:, 4], reduction='sum')
        loss_object = (loss_obj + self.lambda_noobj * loss_noobj) / bs

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_preds, class_targets, reduction='sum') / bs
        # print(loss_boxes, loss_object, loss_class)
        losses = {
            'loss_boxes': loss_boxes,
            'loss_object': loss_object,
            'loss_class': loss_class,
        }

        return losses

    def _transform_targets(self, targets):
        """
        inputs (images): List[Tensor]
        targets: Optional[List[Dict[str, Tensor]]]

        make a copy of targets to avoid modifying it in-place
        targets = [{k: v for k,v in t.items()} for t in targets]
        """
        _check_targets(targets)

        if targets is not None:
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                target: Dict[str, Tensor] = {}
                for k, v in t.items():
                    target[k] = v
                targets_copy.append(target)
            targets = targets_copy

        # STEP 1. Normalized between 0 and 1, 
        # STEP 2. encode a target (boxes, labels)
        # Center x, y, w h는 데이터셋에서 변환해라
        # List[Dict[str, Tensor]]
        # boxes: Tensor
        # labels: Tensor      
        nb = self.num_boxes
        gs = self.grid_size  # grid size
        nc = self.num_classes
        cell_size = 1.0 / gs
        w, h = yolov1_config.max_size  # Tuple[int, int]
        
        # 모형에서 나온 아웃풋과 동일한 모양으로 변환
        # x1, y1, x2, y2를 center x, center y, w, h로 변환하고
        # 모든 0~1사이로 변환, cx, cy는 each cell안에서의 비율
        # w, h는 이미지 대비 비율
        transformed_targets = torch.zeros(len(targets), gs, gs, 5*nb+nc, device=self.device)
        for b, target in enumerate(targets):
            boxes = target['boxes']
            norm_boxes = boxes / torch.Tensor([[w, h, w, h]]).expand_as(boxes).to(self.device)

            xys = (norm_boxes[:, :2] + norm_boxes[:, 2:]) / 2.0
            whs = norm_boxes[:, 2:] - norm_boxes[:, :2]
    
            for box_id in range(boxes.size(0)):
                xy = xys[box_id]
                wh = whs[box_id]
    
                ij = (xy / cell_size).ceil() - 1.0
                top_left = ij * cell_size
                norm_xy = (xy - top_left) / cell_size
                
                i, j = int(ij[0]), int(ij[1])
                for k in range(0, 5*nb, 5):
                    transformed_targets[b, j, i, k:k+4] = torch.cat([norm_xy, wh])
                    transformed_targets[b, j, i, k+4] = 1.0
                
                indices = torch.as_tensor(target['labels'][box_id], dtype=torch.int64).view(-1, 1)
                labels = torch.zeros(indices.size(0), self.num_classes).scatter_(1, indices, 1)
                transformed_targets[b, j, i, 5*nb:] = labels.squeeze()
            
        # print(transformed_targets[0][3])
        return transformed_targets


                


            
def compute_iou(bbox1, bbox2):
    """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
    Args:
        bbox1: (Tensor) bounding bboxes, sized [N, 4].
        bbox2: (Tensor) bounding bboxes, sized [M, 4].
    Returns:
        (Tensor) IoU, sized [N, M].
    """
    N = bbox1.size(0)
    M = bbox2.size(0)

    # Compute left-top coordinate of the intersections
    lt = torch.max(
        bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
        bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    )
    # Conpute right-bottom coordinate of the intersections
    rb = torch.min(
        bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
        bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    )
    # Compute area of the intersections from the coordinates
    wh = rb - lt   # width and height of the intersection, [N, M, 2]
    wh[wh < 0] = 0 # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

    # Compute area of the bboxes
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
    area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
    area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

    # Compute IoU from the areas
    union = area1 + area2 - inter # [N, M, 2]
    iou = inter / union           # [N, M, 2]

    return iou
        




    #     pass


    # def __not_imple(self):

    #     