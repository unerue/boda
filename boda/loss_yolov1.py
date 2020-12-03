import sys

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional

from .loss_base import LoseFunction
# from ..utils import jaccard
from .box_utils import xyxy_to_cxywh
from .box_utils import jaccard


class Match:
    def __init__(self, thresh) -> None:
        self.thresh = thresh

    def __call__(self, preds, targets) -> Any:
        overlaps = jaccard(preds, targets)
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        return best_truth_overlap, best_truth_idx


class Yolov1Loss(LoseFunction):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def encode(self, targets):
        cell_size = 1.0 / self.config.grid_size
        w, h = self.config.max_size
        for target in targets:
            boxes = target['boxes']
            boxes /= torch.Tensor([[w, h, w, h]]).expand_as(boxes)
            boxes = xyxy_to_cxywh(boxes)
            ij = (boxes[:, :2] / cell_size).ceil() - 1.0
            x0y0 = ij * cell_size
            boxes[:, :2] = (boxes[:, :2] - x0y0) / cell_size
            target['boxes'] = boxes

        print('CAT!!')
        print(torch.cat([t['boxes'] for t in targets], dim=0))
        print()

        return targets

    def forward(self, inputs, targets):
        """
            inputs (Dict[str, Tensor])
            targets (Dict[str, Tensor])
        """
        self.check_targets(targets)
        targets = self.copy_targets(targets)
        targets = self.encode(targets)

        gs = self.config.grid_size
        for pred, target in zip(inputs['boxes'], targets):
            pred_xyxy = torch.FloatTensor(pred.size())
            pred_xyxy[:, :2] = pred[:, :2] / gs - 0.5 * pred[:, 2:]
            pred_xyxy[:, 2:] = pred[:, :2] / gs + 0.5 * pred[:, 2:]

            true = target['boxes']
            true_xyxy = torch.FloatTensor(true.size())
            true_xyxy[:, :2] = true[:, :2] / gs - 0.5 * true[:, 2:]
            true_xyxy[:, 2:] = true[:, :2] / gs + 0.5 * true[:, 2:]
            

            print(pred)
            print(true)
            print()
            best_true, best_idx = Match(0.2)(pred_xyxy, true_xyxy)
            print(best_true, best_idx)
            print()
            print(pred[best_idx])
            print()
            pred = pred[best_idx].squeeze(0)
            # print(pred.squeeze(0))
            print(F.mse_loss(pred[:,:2], true[:,:2], reduction='sum'))


        

# class Yolov1Loss(LoseFunction):
#     def __init__(self, config):
#         super().__init__()
#         self.num_boxes = config.num_boxes
#         self.num_classes = config.num_classes
#         self.grid_size = config.grid_size

#         self.lambda_coord = config.lambda_coord
#         self.lambda_noobj = config.lambda_noobj

#     def _transform_targets(self, targets):
#         """
#         inputs (images): List[Tensor]
#         targets: Optional[List[Dict[str, Tensor]]]

#         make a copy of targets to avoid modifying it in-place
#         targets = [{k: v for k,v in t.items()} for t in targets]
#         """
#         self._check_targets(targets)
#         targets = self._copy_target(targets)
#         # STEP 1. Normalized between 0 and 1, 
#         # STEP 2. encode a target (boxes, labels)
#         # Center x, y, w h는 데이터셋에서 변환해라
#         # List[Dict[str, Tensor]]
#         # boxes: Tensor
#         # labels: Tensor   
#         nb = self.num_boxes
#         gs = self.grid_size
#         nc = self.num_classes
#         cell_size = 1.0 / gs
#         w, h = self.config.max_size  # Tuple[int, int]
#         # 모형에서 나온 아웃풋과 동일한 모양으로 변환
#         # x1, y1, x2, y2를 center x, center y, w, h로 변환하고
#         # 모든 0~1사이로 변환, cx, cy는 each cell안에서의 비율
#         # w, h는 이미지 대비 비율
#         transformed_targets = torch.zeros((len(targets), gs, gs, 5*nb+nc), device=self.device)
#         # cnt = 0
        
#         for b, target in enumerate(targets):
#             boxes = target['boxes']
#             norm_boxes = boxes / torch.Tensor([[w, h, w, h]]).expand_as(boxes).to(self.device)
#             # 데이터셋에서 변환해서 들어오기
#             # center x, y, width and height
#             xys = (norm_boxes[:, 2:] + norm_boxes[:, :2]) / 2.0
#             whs = norm_boxes[:, 2:] - norm_boxes[:, :2]
#             for box_id in range(boxes.size(0)):
#                 xy = xys[box_id]
#                 wh = whs[box_id]
                
#                 ij = (xy / cell_size).ceil() - 1.0
#                 i, j = int(ij[0]), int(ij[1])
                
#                 x0y0 = ij * cell_size
#                 norm_xy = (xy - x0y0) / cell_size                
#                 for k in range(0, 5*nb, 5):
#                     if transformed_targets[b, j, i, k+4] == 1.0:
#                         transformed_targets[b, j, i, k+5:k+5+4] = torch.cat([norm_xy, wh])
#                     else:
#                         transformed_targets[b, j, i, k:k+4] = torch.cat([norm_xy, wh])
#                         transformed_targets[b, j, i, k+4] = 1.0
#                     # transformed_targets[b, j, i, k:k+4] = torch.cat([norm_xy, wh])
#                     # transformed_targets[b, j, i, k+4] = 1.0
#                 # print(transformed_targets[b, j, i, :10])
#                 indices = torch.as_tensor(target['labels'][box_id], dtype=torch.int64).view(-1, 1)
#                 labels = torch.zeros(indices.size(0), self.num_classes).scatter_(1, indices, 1)
#                 transformed_targets[b, j, i, 5*nb:] = labels.squeeze()
        
#         return transformed_targets
    
#     def forward(self, inputs: List[Tensor], targets: List[Dict[str, Tensor]]) -> Tensor:
#         """
#         S * S * (B * 5 + C) Tensor
#         where S denotes grid size and for each grid cell predicts B bounding boxes, 
#         confidence for those boxes, and C class probabilites.
#         """
#         if self.training and targets is None:
#             raise ValueError
        
#         self.device = inputs.device
#         bs = inputs.size(0)  # batch size
#         nb = self.num_boxes  # B
#         gs = inputs.size(2)  # S
#         # -> torch.Size([batch_size, S, S, 5*B+C])
#         targets = self._transform_targets(targets)
#         # print(inputs.size(), targets.size())
#         # print(inputs.device, targets.device)
        
#         # -> torch.Size([batch_size, S, S, 5*B+C])
#         coord_mask = targets[..., 4] > 0
#         noobj_mask = targets[..., 4] == 0
#         coord_mask = coord_mask.unsqueeze(-1).expand_as(targets)#.to(self.device)
#         noobj_mask = noobj_mask.unsqueeze(-1).expand_as(targets)#.to(self.device)
#         # print(coord_mask.size(), noobj_mask.size())
#         # print(coord_mask.dtype, noobj_mask.dtype)
#         # print(coord_mask.device, noobj_mask.device)
        
#         # Covert predicted outputs
#         # N: num_bboxes in targets (ground truth) // num_boxes
#         # Example: 총 박스 개수에서 좌표가 중복된 거 제외하고 나머지, 
#         # 좌표가 중복된건 두번째 박스에 집어넣음. github의 다른 코드들은 그렇게 처리하지 않음...
#         # coord_preds -> torch.Size([N, 30]) 
#         # boxes_preds -> torch.Size([N*nb, 5]), [N*nb, [cx, cy, w, h, conf]]
#         # class_preds -> torch.Size([N, 20])
#         coord_preds = inputs[coord_mask].view(-1, 5*nb+self.num_classes)
#         boxes_preds = coord_preds[:, :5*nb].contiguous().view(-1, 5) 
#         class_preds = coord_preds[:, 5*nb:]
#         # print(coord_preds.size(), boxes_preds.size(), class_preds.size())
#         # print(coord_preds.dtype, boxes_preds.dtype, class_preds.dtype)
#         # print(coord_preds.device, boxes_preds.device, class_preds.device)
#         print(boxes_preds[:5])
#         coord_targets = targets[coord_mask].view(-1, 5*nb+self.num_classes)
#         boxes_targets = coord_targets[:, :5*nb].contiguous().view(-1, 5)
#         class_targets = coord_targets[:, 5*nb:]
#         # print(coord_targets.size(), boxes_targets.size(), class_targets.size())
#         # print(coord_targets.dtype, boxes_targets.dtype, class_targets.dtype)
#         # print(coord_targets.device, boxes_targets.device, class_targets.device)
#         print(boxes_targets[:5])
#         # N = 전체 박스에서 물체가 있는 것을 제외한 나머지들
#         # noobj_preds: torch.Size([N, 1])
#         # noobj_preds = [0.534, 0.512, 0.312,...,0.123]
#         # noobj_targets = [0., 0., 0.,...,0.]
#         noobj_preds = inputs[noobj_mask].view(-1, 5*nb+self.num_classes)
#         noobj_targets = targets[noobj_mask].view(-1, 5*nb+self.num_classes)
#         noobj_conf_mask = torch.BoolTensor(noobj_preds.size()).fill_(0).to(self.device)
#         # print(noobj_preds.size(), noobj_targets.size(), noobj_conf_mask.size())
#         # print(noobj_preds.dtype, noobj_targets.dtype, noobj_conf_mask.dtype)
#         # print(noobj_preds.device, noobj_targets.device, noobj_conf_mask.device)
#         for b in range(nb):
#             noobj_conf_mask[:, 4+b*5] = 1
        
#         noobj_conf_preds = noobj_preds[noobj_conf_mask]
#         noobj_conf_targets = noobj_targets[noobj_conf_mask]
#         # print(noobj_conf_preds.size(), noobj_conf_targets.size())
#         # print(noobj_conf_preds.dtype, noobj_conf_targets.dtype)
#         # Compute loss for the cells with objects.
#         loss_noobj = F.mse_loss(noobj_conf_preds, noobj_conf_targets, reduction='sum')
                
#         coord_response_mask = torch.BoolTensor(boxes_targets.size()).fill_(0).to(self.device)
#         coord_not_response_mask = torch.BoolTensor(boxes_targets.size()).fill_(1).to(self.device)
#         # print(coord_response_mask.dtype, coord_not_response_mask.dtype)
#         # print(coord_response_mask.device, coord_not_response_mask.device)
#         # print(coord_response_mask.size(), coord_not_response_mask.size())

#         # torch.Size([N, 5]) only the last column is used
#         iou_targets = torch.zeros(boxes_targets.size(), device=self.device)
#         # Choose the predicted bbox having the highest IoU for each target bbox.
#         # 박스 두개씩 응답된 iou가 큰 박스는 1, 아니면 0, 두개씩 비교
#         # iou return [0.031, 0.512]
#         for i in range(0, boxes_targets.size(0), 2):
#             _preds = boxes_preds[i:i+nb]
#             xyxy_preds = torch.FloatTensor(_preds.size()).to(self.device)
#             # normalized for cell-size and image-size respectively,
#             # rescale (center_x,center_y) for the image-size to compute IoU correctly.
#             xyxy_preds[:, 0:2] = _preds[:, :2]/gs - 0.5 * _preds[:, 2:4]
#             xyxy_preds[:, 2:4] = _preds[:, :2]/gs + 0.5 * _preds[:, 2:4]
#             # target bbox at i-th cell. Because target boxes contained by each cell 
#             # are identical in current implementation, enough to extract the first one.
#             _targets = boxes_targets[i].view(-1, 5)
#             # return [x1, y1, x2, y2, conf]
#             xyxy_targets = torch.FloatTensor(_targets.size()).to(self.device)
#             # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for 
#             # cell-size and image-size respectively,
#             # rescale (center_x,center_y) for the image-size to compute IoU correctly.
#             xyxy_targets[:, 0:2] = _targets[:, :2]/gs - 0.5 * _targets[:, 2:4]
#             xyxy_targets[:, 2:4] = _targets[:, :2]/gs + 0.5 * _targets[:, 2:4]
#             # iou = jaccard(xyxy_preds[..., :4], xyxy_targets[..., :4]) # [B, 1]
#             iou = compute_iou(xyxy_preds[..., :4], xyxy_targets[..., :4])
#             max_iou, max_index = iou.max(0)
#             # print(max_iou.dtype, max_index.dtype)
#             # print(max_iou.device, max_index.device)
#             coord_response_mask[i+max_index] = 1
#             coord_not_response_mask[i+max_index] = 0
#             # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
#             # from the original paper of YOLO.
#             iou_targets[i+max_index, 4] = max_iou
        
#         # BBox location/size and objectness loss for the response bboxes.
#         response_boxes_preds = boxes_preds[coord_response_mask].view(-1, 5)
#         response_boxes_targets = boxes_targets[coord_response_mask].view(-1, 5)
#         iou_targets = iou_targets[coord_response_mask].view(-1, 5)
#         # print(response_boxes_preds.size(), response_boxes_targets.size(), iou_targets.size())
#         # print(response_boxes_preds.dtype, response_boxes_targets.dtype, iou_targets.dtype)

#         loss_xy = F.mse_loss(
#             response_boxes_preds[:, :2], 
#             response_boxes_targets[:, :2], reduction='sum')

#         # loss_xy = torch.sum(torch.pow(response_boxes_targets[:, 0] - response_boxes_preds[:, 0], 2) \
#         # + torch.pow(response_boxes_targets[:, 1] - response_boxes_preds[:, 1], 2))

#         loss_wh = F.mse_loss(
#             torch.sqrt(response_boxes_preds[:, 2:4]), 
#             torch.sqrt(response_boxes_targets[:, 2:4]), reduction='sum')

#         # loss_wh = torch.sum(torch.pow(torch.sqrt(response_boxes_targets[:, 3]) - torch.sqrt(response_boxes_preds[:, 3]), 2) \
#         # + torch.pow(torch.sqrt(response_boxes_targets[:, 4]) - torch.sqrt(response_boxes_preds[:, 4]), 2))
        
#         loss_boxes = (self.lambda_coord * (loss_xy + loss_wh)) / bs 
#         # loss_boxes = (self.lambda_coord * (loss_xy)) / bs 

#         loss_obj = F.mse_loss(
#             response_boxes_preds[:, 4], iou_targets[:, 4], reduction='sum')
#         loss_object = (loss_obj + (self.lambda_noobj * loss_noobj)) / bs
#         # Class probability loss for the cells which contain objects.
#         loss_class = F.mse_loss(class_preds, class_targets, reduction='sum') / bs

#         losses = {
#             'loss_boxes': loss_boxes,
#             'loss_object': loss_object,
#             'loss_class': loss_class,
#         }

#         return losses



               


            
# def compute_iou(bbox1, bbox2):
#     """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
#     Args:
#         bbox1: (Tensor) bounding bboxes, sized [N, 4].
#         bbox2: (Tensor) bounding bboxes, sized [M, 4].
#     Returns:
#         (Tensor) IoU, sized [N, M].
#     """
#     N = bbox1.size(0)
#     M = bbox2.size(0)

#     # Compute left-top coordinate of the intersections
#     lt = torch.max(
#         bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
#         bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
#     )
#     # Conpute right-bottom coordinate of the intersections
#     rb = torch.min(
#         bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
#         bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
#     )
#     # Compute area of the intersections from the coordinates
#     wh = rb - lt   # width and height of the intersection, [N, M, 2]
#     wh[wh < 0] = 0 # clip at 0
#     inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

#     # Compute area of the bboxes
#     area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
#     area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
#     area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
#     area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

#     # Compute IoU from the areas
#     union = area1 + area2 - inter # [N, M, 2]
#     iou = inter / union           # [N, M, 2]

#     return iou
        


