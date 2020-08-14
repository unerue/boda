import sys

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional

from .loss_base import _check_targets
from ..configuration import yolov1_config
from ..utils import intersect, jaccard
# from torch.jit.annotations import Tuple, List, Dict, Any, Optional


class Yolov1Loss(nn.Module):
    def __init__(self, lambda_coord=0.5, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = yolov1_config.lambda_coord
        self.lambda_noobj = yolov1_config.lambda_noobj
        self.num_boxes = yolov1_config.num_boxes
        self.num_classes = yolov1_config.dataset.num_classes
        self.grid_size = yolov1_config.grid_size

    def forward(self, inputs: List[Tensor], targets: List[Dict[str, Tensor]]) -> Tensor:
        if self.training and targets is None:
            raise ValueError
        
        nb = self.num_boxes
        device = inputs.device
        bs = inputs.size(0)
        gs = inputs.size(2)

        targets = self._transform_targets(targets)
        print(inputs.size(), targets.size())

        coord_mask = targets[..., 4] > 0
        noobj_mask = targets[..., 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(targets)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(targets)
        print(coord_mask.dtype, noobj_mask.dtype)
        print(coord_mask.device, noobj_mask.device)

        coord_pred = inputs[coord_mask].view(-1, 30)  
        boxes_pred = coord_pred[:, :5*nb].contiguous().view(-1, 5)    # [n_coord x B, 5=len([x, y, w, h, conf])]
        label_pred = coord_pred[:, 5*nb:]
        print(coord_pred.dtype, boxes_pred.dtype)
        print(coord_pred.device, boxes_pred.device)

        coord_target = targets[coord_mask].view(-1, 5*nb+self.num_classes)
        boxes_target = coord_target[:, :5*nb].contiguous().view(-1, 5)# [n_coord x B, 5=len([x, y, w, h, conf])]
        label_target = coord_target[:, 5*nb:]


        # Compute loss for the cells with no object bbox.
        # pred tensor on the cells which do not contain objects. [n_noobj, N]
        # n_noobj: number of the cells which do not contain objects.
        noobj_pred = inputs[noobj_mask].view(-1, 5*nb+self.num_classes)        
        
        # target tensor on the cells which do not contain objects. [n_noobj, N]
        # n_noobj: number of the cells which do not contain objects.
        noobj_target = targets[noobj_mask].view(-1, 5*nb+self.num_classes).to(device)
        # [n_noobj, N]
        noobj_conf_mask = torch.BoolTensor(noobj_pred.size()).fill_(0).to(device)
        print(noobj_conf_mask.dtype, noobj_target.dtype)
        print(noobj_conf_mask.device, noobj_target.device)

        for b in range(2):
            noobj_conf_mask[:, 4+b*5] = 1 # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1

        noobj_pred_conf = noobj_pred[noobj_conf_mask]       # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]   # [n_noobj, 2=len([conf1, conf2])]
        print(noobj_pred_conf.device, noobj_target_conf.device)

        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        # [n_coord x B, 5] # [n_coord x B, 5]
        coord_response_mask = torch.BoolTensor(boxes_target.size()).fill_(0)    
        coord_not_response_mask = torch.BoolTensor(boxes_target.size()).fill_(1)
        print(coord_response_mask.dtype, coord_not_response_mask.dtype)
        print(coord_response_mask.device, coord_not_response_mask.device)
        print(coord_response_mask.size(), coord_not_response_mask.size())
        
        # [n_coord x B, 5], only the last 1=(conf,) is used
        bbox_target_iou = torch.zeros(boxes_target.size(), device=device)

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, boxes_target.size(0), 2):
            # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred = boxes_pred[i:i+2] 
            # [B, 5=len([x1, y1, x2, y2, conf])]
            pred_xyxy = torch.FloatTensor(pred.size())
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.

            pred_xyxy[:,  :2] = pred[:, :2]/float(gs) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2]/float(gs) + 0.5 * pred[:, 2:4]

            target = boxes_target[i] # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = boxes_target[i].view(-1, 5) # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = torch.FloatTensor(target.size()) # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:,  :2] = target[:, :2]/float(gs) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2]/float(gs) + 0.5 * target[:, 2:4]

            iou = jaccard(pred_xyxy[..., :4], target_xyxy[..., :4]) # [B, 1]
            

            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        
        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = boxes_pred[coord_response_mask].view(-1, 5).to('cuda')      # [n_response, 5]
        bbox_target_response = boxes_target[coord_response_mask].view(-1, 5).to('cuda')  # [n_response, 5], only the first 4=(x, y, w, h) are used

        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5).to('cuda')        # [n_response, 5], only the last 1=(conf,) is used

        loss_xy = F.mse_loss(bbox_pred_response[:, :2].to('cuda'), bbox_target_response[:, :2].to('cuda'), reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4].to('cuda')), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4].to('cuda'), target_iou[:, 4], reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(label_pred.to('cuda'), label_target.to('cuda'), reduction='sum')

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(bs)

        losses = {
            'loss_classifier': None,
            'loss_boxes': None,
            'loss_class': None,
            'loss_xy': None,
            'loss_wh': None,
            'loss_obj': None,
        }

        return loss

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

        transformed_targets = torch.zeros(len(targets), gs, gs, 5*nb+nc)

        # Tuple[int, int]
        w, h = yolov1_config.max_size
        for b, target in enumerate(targets):
            boxes = target['boxes']
            norm_boxes = boxes / torch.Tensor([[w, h, w, h]]).expand_as(boxes).to(boxes.device)

            cxys = (norm_boxes[:, 2:] + norm_boxes[:, :2]) / 2
            whs = norm_boxes[:, 2:] - norm_boxes[:, :2]

            for box_id in range(boxes.size(0)):
                cxy = cxys[box_id]
                wh = whs[box_id]
                ij = (cxy / cell_size).ceil() - 1.0
                top_left = ij * cell_size
                norm_xy = (cxy - top_left) / cell_size
                i, j = int(ij[0]), int(ij[1])

                for k in range(0, 5*2, 5):
                    transformed_targets[b, j, i, k:k+4] = torch.cat([norm_xy, wh])
                    transformed_targets[b, j, i, k+4] = 1.0

                transformed_targets[b, j, i, 5*2:] = target['labels'][box_id]

        return transformed_targets


                


            

        




    #     pass


    # def __not_imple(self):

    #     