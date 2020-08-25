import sys
import time
import math
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..backbone import darknet53, Shortcut
# from ..configuration import yolov3_base_darknet_pascal
from .architecture_base import Conv2d1x1, Upsample
# from .loss_yolov3 import Yolov3Loss


class Yolov3PredictionHead(nn.Module):
    """Prediction Head for YOLOv3.

    백본에서 아웃풋을 받아서 로스 펑션에 넣을 수 있게 변환해주는 헤드
    pretrained = False
    """
    def __init__(self, config: Dict = None):
        super().__init__()
        # TODO: 모든 constant -> config 처리
        self.config = config
        # self.backbone = darknet53()  # self.config.backbone
        
        out_channels = self.backbone.channels[-self.config.selected_layers:]  
        # B = 3, C = 20
        # N × N × [B ∗ (4 + 1 + C)]
        num_features = self.config.num_boxes * (4+1+self.config.dataset.num_classes)

        # 마지막 고정 레이어
        self.detect_layers = nn.ModuleList()
        for in_channels in out_channels:
            self.detect_layers.append(Conv2d1x1(in_channels, num_features))

        # 업샘플링
        self.upsample_layers = nn.ModuleList()
        for in_channels in out_channels[1:]:
            self.upsample_layers.append(
                nn.Sequential(
                    Conv2d1x1(in_channels, int(in_channels*0.25)),
                    Upsample(2)))

        # 기본 레이어 residual X
        self.shortcut_layers = nn.ModuleList()
        for in_channels, downsample in zip(out_channels[1:], self.config.downsample):
            _in_channels = [downsample[1]] * (self.config.selected_layers-1)
            _in_channels.insert(0, in_channels - int(in_channels*0.25))
            self.shortcut_layers.append(
                nn.Sequential(*[Shortcut(_in, downsample, residual=False) for _in in _in_channels]))

    def forward(self, inputs: torch.Tensor):
        features = self.backbone(inputs)[-self.config.selected_layers:]
        j = len(features)
        outs = []
        for feature in reversed(features):
            if j < len(features):
                concat_out = torch.cat([feature, residual], 1)
                # print(concat_out.size(), j, '** CONCATENATE!!')
                features = self.shortcut_layers[j-1](concat_out)
                # print(out.size(), j, '** CONCAT + DETECTION LAYER!!!')
            j -= 1
            # if j != 0:
            out = self.detect_layers[j](feature)
            # print(out.size(), j, '******* DETECTION !!!!')
            if j != 0:
                residual = self.upsample_layers[j-1](feature)
                # print(residual.size(), j, '** UPSAMPLE!!')
            outs.append(out)
            
        return outs


class Yolov3Model(nn.Module):
    """YOLOv3 architecture

    Yolov3PredictionHead ->
    Yolov3Loss(images, targets)
    참고
        [1] 토치비전 -> Mask R-CNN 구조
        [2] huggingface
        [3]

    out = model(images, targets)
    losses = sum(loss for loss in out.values())
    
    """
    def __init__(self, config, num_classes=20):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.transform = None
        self.prediction_head = Yolov3PredictionHead(config)
        # print(self.prediction_head.backbone)

    def forward(self, inputs):
        # inputs = self.transform(inputs)
        # if self.training
    
        inputs = torch.stack(inputs)
        # inputs = torch.FloatTensor(inputs)
        outs = self.prediction_head(inputs)
        preds = []
        for out, mask in zip(outs, self.config.masks):
            # batch_size = out.size(0)
            grid_size = out.size(2)
            anchors = [self.config.anchors[i] for i in mask]
            self._generate_grid_anchors(grid_size, anchors)
  
            out = out.view(out.size(0), 3, self.num_classes+5, out.size(2), out.size(2)).permute(0, 1, 3, 4, 2).contiguous()
            
            x = torch.sigmoid(out[..., 0])  # predicted center xs
            y = torch.sigmoid(out[..., 1])  # predicted center ys
            w = out[..., 2]  # predicted widths
            h = out[..., 3]  # predicted heights

            boxes = torch.zeros_like(out[..., :4])
            boxes[..., 0] = x + self.grid_x
            boxes[..., 1] = y + self.grid_y
            boxes[..., 2] = torch.exp(w) * self.anchor_w
            boxes[..., 3] = torch.exp(h) * self.anchor_h
            boxes = boxes * self.stride
           
            scores = torch.sigmoid(out[..., 4])  # predicted confidences
            labels = torch.sigmoid(out[..., 5:])  # predicted labels
           
            preds.append({
                'boxes': boxes,  # [x, y, w, h]
                'scores': scores,  # confidence scores
                'labels': labels,  # predict_proba
                'scaled_anchors': self.scaled_anchors,
            })
        # TODO: postprocessing 후처리 시스템 만들기 if not self.training: 바로 아웃풋 나올수 있는
 
        return preds

    def _generate_grid_anchors(self, grid_size, anchors, device: str = 'cpu'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # 삭제해라!!!!
        num_anchors = len(anchors)
        self.stride = self.config.max_size / grid_size
        self.grid_x = torch.arange(grid_size, device=device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        self.grid_y = torch.arange(grid_size, device=device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])    
        
        self.scaled_anchors = torch.tensor([
            (aw/self.stride, ah/self.stride) for aw, ah in anchors], device=device)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))



       



    # def _grid_offset(self, grid_size, anchors):
    #     self.stride = self.config.max_size / grid_size

    #     self.grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
    #     self.grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])
        
    #     num_anchors = len(anchors)
    #     self.scaled_anchors = torch.FloatTensor([
    #         (aw/self.stride, ah/self.stride) for aw, ah in anchors])

    #     self.anchor_w = self.scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
    #     self.anchor_h = self.scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))


# def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
#     nB = pred_boxes.size(0)
#     nA = pred_boxes.size(1)
#     nC = pred_cls.size(-1)
#     nG = pred_boxes.size(2)

#     # Output tensors
#     obj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
#     noobj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(1)
#     class_mask = torch.FloatTensor(nB, nA, nG, nG).fill_(0)

#     iou_scores = torch.FloatTensor(nB, nA, nG, nG).fill_(0)

#     tx = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
#     ty = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
#     tw = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
#     th = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
#     tcls = torch.FloatTensor(nB, nA, nG, nG, nC).fill_(0)

#     # Convert to position relative to box
#     target_boxes = target[:, 2:6] * nG
#     gxy = target_boxes[:, :2]
#     gwh = target_boxes[:, 2:]
#     # Get anchors with best iou
#     ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
#     best_ious, best_n = ious.max(0)
#     # Separate target values
#     b, target_labels = target[:, :2].long().t()
#     gx, gy = gxy.t()
#     gw, gh = gwh.t()
#     gi, gj = gxy.long().t()
#     # Set masks
#     obj_mask[b, best_n, gj, gi] = 1
#     noobj_mask[b, best_n, gj, gi] = 0

#     # Set noobj mask to zero where iou exceeds ignore threshold
#     for i, anchor_ious in enumerate(ious.t()):
#         noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

#     # Coordinates
#     tx[b, best_n, gj, gi] = gx - gx.floor()
#     ty[b, best_n, gj, gi] = gy - gy.floor()
#     # Width and height
#     tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
#     th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
#     # One-hot encoding of label
#     tcls[b, best_n, gj, gi, target_labels] = 1
#     # Compute label correctness and iou at best anchor
#     class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
#     iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

#     tconf = obj_mask.float()
#     return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf