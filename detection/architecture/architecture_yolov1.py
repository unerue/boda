import sys
import time
import math
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..backbone import darknet9, darknet21
from .architecture_base import _check_inputs



from torchvision import datasets, models, transforms
from torchsummary import summary



backbone1 = models.resnet50(pretrained=True)
# print(summary(backbone1, (3, 448, 448)))
# print(backbone1.layer4)
for p in backbone1.parameters():
    p.requires_grad_ = False
# sys.exit(0)


class Yolov1PredictionHead(nn.Module):
    def __init__(self, config: Dict, backbone: nn.Module = darknet9()):
        super().__init__()
        self.num_classes = config.dataset.num_classes
        self.grid_size = config.grid_size
        self.num_boxes = config.num_boxes
        # self.backbone = backbone
        # self.backbone = darknet21()
        self.backbone = backbone1
        
        self.out_channels = 5 * self.num_boxes + self.num_classes
        # self.fc = nn.Linear(256*3*3, 1470)
        self.fc = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1))

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 7 * 7 * self.out_channels),
            nn.Sigmoid())

    def forward(self, x: List[Tensor]) -> Tensor:
        """
        """
        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)



        # x = self.fc(torch.flatten(x[-1], 1))
        # x = x.view(-1, self.grid_size, self.grid_size, self.out_channels)
        # x = self.fc1(x[-1])
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # print(x.size())
        x = x.view(-1, self.grid_size, self.grid_size, self.out_channels)
        
        return x


class Yolov1Model(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.max_size = (448, 448)
        self.prediction_head = Yolov1PredictionHead(config)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        """
        Arguments
        ------
            inputs: List[torch.Tensor] 
        
        Returns
        -------
            preds: List[Dict[torch.Tensor, torch.Tensor, torch.Tensor]]
                boxes: torch.Size([n, 4])
                scores: torch.Size([n, 1])
                labels: torch.Size([n, 20])
        """
        # images = [image for image in inputs]
        _check_inputs(inputs)

        # if batch size 1,
        if not isinstance(inputs, Tensor):
            inputs = torch.stack(inputs)
        
        if self.training:
            outputs = self.prediction_head(inputs)

            # self.device = outputs.device
            # gs = self.config.grid_size
            # nb = self.config.num_boxes
            
            # cell_size = 1./gs
            
            # grid_x = torch.arange(gs, device=self.device).repeat(gs, 1).view(1, 1, gs, gs)
            # grid_y = torch.arange(gs, device=self.device).repeat(gs, 1).t().view(1, 1, gs, gs)

            # norm_x = grid_x * cell_size 
            # norm_y = grid_y * cell_size

            # print(outputs[0,..., :4])
            # outputs[..., 0] = (outputs[..., 0] + grid_x )  / cell_size
            # outputs[..., 1] = (outputs[..., 1] + grid_x ) / cell_size

            # outputs[..., 5] = outputs[..., 5] * cell_size + norm_x
            # outputs[..., 6] = outputs[..., 6] * cell_size + norm_y
            # print('='*100)
            # print(outputs[0,..., :4])

            # norm_xy = outputs[..., 0:2]
            # norm_wh = outputs[..., 2:4]
            # outputs[..., 2:4] = norm_xy + 0.5 * norm_wh

            # norm_xy = outputs[..., 5:7]
            # norm_wh = outputs[..., 7:9]
            # outputs[..., 7:9] = norm_xy + 0.5 * norm_wh
            # print('='*100)
            # print(outputs[0,..., :4].half())

            # sys.exit(0)

            # for i in range(0, 5*nb, 5):
            #     # torch.Size([2, 7, 7, 4]) -> [[[x,y,x,y]], [[...]]]
            #     boxes = outputs[..., i:i+4] 
            #     # torch.Size([2, 7, 7]) -> [[[]], [[]]] 
            #     scores = outputs[..., i+5]  
            #     # torch.Size([2, 7, 7])
            #     mask = (scores * labels_proba) > 0.5 
            #     if not mask.size(0):
            #         continue
            
            #     boxes[..., 0] = boxes[..., 0] * cell_size + norm_x
            #     boxes[..., 1] = boxes[..., 1] * cell_size + norm_y

            #     norm_xy = boxes[..., :2]
            #     norm_wh = boxes[..., 2:]

            #     boxes[..., :2] = norm_xy - 0.5 * norm_wh
            #     boxes[..., 2:] = norm_xy + 0.5 * norm_wh

            # nb = self.num_boxes
            # gs = self.grid_size  # grid size
            # nc = self.num_classes
            # cell_size = 1.0 / gs
            # w, h = yolov1_config.max_size  # Tuple[int, int]
            
            # # 모형에서 나온 아웃풋과 동일한 모양으로 변환
            # # x1, y1, x2, y2를 center x, center y, w, h로 변환하고
            # # 모든 0~1사이로 변환, cx, cy는 each cell안에서의 비율
            # # w, h는 이미지 대비 비율
            # transformed_targets = torch.zeros(len(targets), gs, gs, 5*nb+nc, device=self.device)
            # for b, target in enumerate(targets):
            #     boxes = target['boxes']
            #     norm_boxes = boxes / torch.Tensor([[w, h, w, h]]).expand_as(boxes).to(self.device)

            #     xys = (norm_boxes[:, :2] + norm_boxes[:, 2:]) / 2.0
            #     whs = norm_boxes[:, 2:] - norm_boxes[:, :2]
        
            #     for box_id in range(boxes.size(0)):
            #         xy = xys[box_id]
            #         wh = whs[box_id]
        
            #         ij = (xy / cell_size).ceil() - 1.0
            #         top_left = ij * cell_size
            #         norm_xy = (xy - top_left) / cell_size
                    
            #         i, j = int(ij[0]), int(ij[1])
            #         for k in range(0, 5*nb, 5):
            #             transformed_targets[b, j, i, k:k+4] = torch.cat([norm_xy, wh])
            #             transformed_targets[b, j, i, k+4] = 1.0
                    
            #         indices = torch.as_tensor(target['labels'][box_id], dtype=torch.int64).view(-1, 1)
            #         labels = torch.zeros(indices.size(0), self.num_classes).scatter_(1, indices, 1)
            #         transformed_targets[b, j, i, 5*nb:] = labels.squeeze()


            return outputs
        else:
            with torch.no_grad():
                outputs = self.prediction_head(inputs)
            
            self.device = outputs.device
            gs = self.config.grid_size
            nb = self.config.num_boxes
            bs = outputs.size(0)
            
            cell_size = 1.0 / gs
            
            # grid_x = torch.arange(gs, device=self.device).repeat(gs, 1).view(1, 1, gs, gs)
            # grid_y = torch.arange(gs, device=self.device).repeat(gs, 1).t().view(1, 1, gs, gs)

            # norm_x = grid_x * cell_size
            # norm_y = grid_y * cell_size
            # print(norm_x.size())
            # print(norm_y.size())

            # preds = []
            # # torch.Size([B, 7, 7, 30]) -> [[[...num_classes]], [[...]]]
            # labels = outputs[..., 5*nb:]
            # # torch.Size([2, 7, 7])
            # labels_proba = outputs[..., 5*nb:].argmax(-1)  
            # for i in range(0, 5*nb, 5):
            #     # torch.Size([2, 7, 7, 4]) -> [[[x,y,x,y]], [[...]]]
            #     boxes = outputs[..., i:i+4] 
            #     # torch.Size([2, 7, 7]) -> [[[]], [[]]] 
            #     scores = outputs[..., i+5]  
            #     # torch.Size([2, 7, 7])
            #     mask = (scores * labels_proba) > 0.2
            #     if not mask.size(0):
            #         continue
            
            #     # x0 = boxes[..., 0] * cell_size + norm_x
            #     # y0 = boxes[..., 1] * cell_size + norm_y
            #     x0 = boxes[..., 0] * cell_size + norm_x
            #     y0 = boxes[..., 1] * cell_size + norm_y

            #     # norm_xy = boxes[..., :2]
            #     norm_wh = boxes[..., 2:]

            #     # [x1, y1, x2, y2]
            #     # boxes[..., :2] = top_left - 0.5 * norm_wh
            #     # boxes[..., 2:] = top_left + 0.5 * norm_wh
            #     boxes[..., 0] = x0 - 0.5 * norm_wh[...,0]
            #     boxes[..., 1] = y0 - 0.5 * norm_wh[...,1]
            #     boxes[..., 2] = x0 + 0.5 * norm_wh[...,0]
            #     boxes[..., 3] = y0 + 0.5 * norm_wh[...,1]

            #     # boxes = torch.cat([boxes[...,0]*w, boxes[...,1]*w, boxes[...,2]*h, boxes[...,3]*h], 1)
                
            #     # boxes = xywh2xyxy(boxes)

            #     preds.append({
            #         'boxes': boxes[mask],
            #         'scores': scores[mask],
            #         'labels': labels[mask]
            #     })\
            print(outputs.size())
            preds = []
            
            for bs in range(outputs.size(0)):
                boxes = []
                scores = []
                labels = []
                for i in range(gs):
                    for j in range(gs):
                        label = outputs[bs, j, i, 5*nb:]
                        class_proba, _ = torch.max(outputs[bs, j, i, 5*nb:], dim=0)
                        for k in range(nb):
                            score = outputs[bs, j, i, k*5+4]
                            proba = score * class_proba
                            proba = score * 1.0
                            if proba < 0.2:
                                continue

                            box = outputs[bs, j, i, k*5:k*5+4]
                            x0y0 = torch.FloatTensor([i,j]).to(self.device) * cell_size

                            norm_xy = box[:2] * cell_size + x0y0
                            norm_wh = box[2:]

                            xyxy = torch.FloatTensor(4).to(self.device)
                            xyxy[:2] = norm_xy - 0.5 * norm_wh
                            xyxy[2:] = norm_xy + 0.5 * norm_wh
                            # print(class_proba.size(), class_label.size())
                            
                            boxes.append(xyxy)
                            scores.append(score)
                        # print(boxes)
                        labels.append(label)
                print(len(boxes))
                if len(boxes) > 0:
                    preds.append({
                        'boxes': torch.stack(boxes),
                        'scores': torch.stack(scores),
                        'labels': torch.stack(labels)})
                else:
                    preds.append({
                        'boxes': torch.zeros((1, 4)),
                        'scores': torch.zeros(1),
                        'labels': torch.zeros((1, 20))})
            
            # print(preds)
            # sys.exit(0)
            preds = self._encode_outputs(preds)

            return preds

    def _decode_outputs(self, inputs):
        return NotImplementedError

    def _encode_outputs(self, inputs):
        # Get detected boxes_detected, labels, confidences, class-scores.
        # Return to self.decode(outputs)
        # boxes_normalized_all, class_labels_all, confidences_all, class_scores_all = self.decode(pred_tensor)
        w, h = self.max_size
        
        # preds = inputs
        # print('*'*100)
        preds = non_maximum_supression(inputs)
        
        # print('*'*100)
        # print(len(preds))
        for pred in preds:
            boxes = pred['boxes']
            # boxes = xywh2xyxy(boxes)
            if boxes.size(0) == 0:
                continue

            boxes[:, 0], boxes[:, 1] = boxes[:, 0]*w, boxes[:, 1]*h
            boxes[:, 2], boxes[:, 3] = boxes[:, 2]*w, boxes[:, 3]*h     

            pred['boxes'] = boxes

        return preds
            # print('='*100)
            # print(boxes.size())
            # print([boxes[...,0]*w, boxes[...,1]*w, boxes[...,2]*h, boxes[...,3]*h])

            
            # print(preds)
            # preds: torch.Size([num_boxes, 4]), torch.Size([num_boxes, 1]), torch.Size([num_boxes, 20])        
            # NMS 후 selected boxes에 담아서 원본 이미지 크기로 다시 재변환
            # selected_boxes = []
            # for pred in preds:
            #     print(len(pred['boxes']))
            #     print(non_maximum_supression(pred['boxes'], pred['scores']))
            #     print(len(non_maximum_supression(pred['boxes'], pred['scores'])))


            # ## Recover inputs image size!!!
            # boxes_detected, class_names_detected, probs_detected = [], [], []
            # for b in range(boxes_normalized.size(0)):
            #     box_normalized = boxes_normalized[b]
            #     class_label = class_labels[b]
            #     prob = probs[b]

            #     x1, x2 = w * box_normalized[0], w * box_normalized[2] # unnormalize x with image width.
            #     y1, y2 = h * box_normalized[1], h * box_normalized[3] # unnormalize y with image height.
            #     boxes_detected.append(((x1, y1), (x2, y2)))

            #     class_label = int(class_label) # convert from LongTensor to int.
            #     class_name = self.class_name_list[class_label]
            #     class_names_detected.append(class_name)

            #     prob = float(prob) # convert from Tensor to float.
            #     probs_detected.append(prob)

        # sys.exit(0)
            

        # outputs[...,0] = outputs[...,0] * (1./7) + norm_x
        # outputs[...,1] = outputs[...,1] * (1./7) + norm_y

        # xy_normal = outputs[...,:2]
        # wh_normal = outputs[...,2:4]

        # outputs[...,:2] = xy_normal - 0.5 * wh_normal
        # outputs[...,2:4] = xy_normal + 0.5 * wh_normal
        
        # print(outputs[...,2:4])

        # outputs = outputs.view(outputs.size(0), 1, 30, outputs.size(2), outputs.size(2)).permute(0, 1, 3, 4, 2).contiguous()
        # print(outputs[0][0].size(), outputs[0][0][0])
        # outputs = outputs.view(outputs.size(0), 7*7, 30)
        # print(outputs.size())
        # print(outputs[0][0])
        # print(outputs[1][0])

        # boxes = outputs[...,:2*5]
        # labels = outputs[...,2*5:]
        # print('before:', boxes[0][0])
        # print('before:', boxes[1][0])
        # print(boxes.size(), labels.size())
        # boxes = boxes.view(-1,5*2)
        # labels = labels.view(-1,20)
        # print(boxes.size(), labels.size())
        # print('after:', boxes[0])
        # print('after:', boxes[49])

        # cell_size = 1./14
        # print(len(boxes))
        # for i in range(len(boxes)):
        #     for j in range(0, 2*5, 5): # num_bboxes B * 5) + C
        #         bbox = boxes[i,j:j+5]
        #         x = bbox[0].item()
        #         y = bbox[1].item()
        #         w = bbox[2]
        #         h = bbox[3]
        #         score = bbox[5]
        #         print(bbox.size(), bbox)
                # boxes[i,:2] = , boxes[i,:2] * cell_size
                # print(boxes)

def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """ Convert [x, y, w, h] to [x_min, y_min, x_max, y_max]
        
    boxes: (B, 1, 4) -> [[[x, y, w, h]]] -> [[[x, y, x, y]]]
    """
    xyxy_boxes = torch.zeros_like(boxes)
    xyxy_boxes[..., 0] = boxes[..., 0] - boxes[..., 2]/2  # x_min
    xyxy_boxes[..., 1] = boxes[..., 1] - boxes[..., 3]/2  # y_min
    xyxy_boxes[..., 2] = boxes[..., 0] + boxes[..., 2]/2  # x_max
    xyxy_boxes[..., 3] = boxes[..., 1] + boxes[..., 3]/2  # y_max

    return xyxy_boxes

# def non_maximum_supression(boxes, scores, threshold=0.5):
def non_maximum_supression(inputs, threshold=0.5):
    """Non-maximum supression
    
    boxes: torch.Size(N, 4)
    scores: torch.Size(N, )
    labels: torch.Size(N, num_classes)
    """
    outputs = []
    for pred in inputs:
        boxes = pred['boxes']
        scores = pred['scores']

        # boxes = xywh2xyxy(boxes)
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)

        _, indices = scores.sort(0, descending=True)

        keeps = []
        while indices.numel():
            i = indices.item() if (indices.numel() == 1) else indices[0].item()
            keeps.append(i)
            
            if indices.numel() == 1:
                break
            
            inter_x1 = x1[indices[1:]].clamp(min=x1[i]) # [m-1, ]
            inter_y1 = y1[indices[1:]].clamp(min=y1[i]) # [m-1, ]
            inter_x2 = x2[indices[1:]].clamp(max=x2[i]) # [m-1, ]
            inter_y2 = y2[indices[1:]].clamp(max=y2[i]) # [m-1, ]

            inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
            inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]
            # intersections b/w/ box `i` and other boxes, sized [m-1, ].
            inters = inter_w * inter_h 
            # unions b/w/ box `i` and other boxes, sized [m-1, ].
            unions = areas[i] + areas[indices[1:]] - inters 
            ious = inters / unions # [m-1, ]
            # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
            # ids_keep = (ious >= threshold).nonzero().squeeze() 
            ids_keep = (ious <= threshold).nonzero().squeeze() 
            if ids_keep.numel() == 0:
                break # If no box left, break.

            indices = indices[ids_keep+1]
        # print('KEEP!!!', keeps)
        # print('KEEP BOX!!!!', boxes[keeps])
        # print(boxes[keeps].size())
        if len(keeps) != 0:
            outputs.append({
                'boxes': boxes[keeps],
                'scores': scores[keeps],
                'labels': pred['labels'][keeps],
            })
        else:
            pass
    # print('+'*100)
    # print(outputs)
    # print('+'*100)
    return outputs


