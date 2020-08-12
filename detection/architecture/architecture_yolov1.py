import sys
import time
import math
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import darknet9
from ..configuration import yolov3_base_darknet_pascal


class Yolov1PredictionHead(nn.Module):
    def __init__(self, config=None, num_classes=20, backbone=darknet9()):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.backbone = backbone
        self.grid_size = 7
        self.out_channels = 5*2+num_classes
        self.fc = nn.Linear(256*3*3, 1470)

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        output = self.fc(torch.flatten(outputs[-1], 1))
        output = output.view(-1, 7, 7, self.out_channels)
        
        return output


class Yolov1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction_head = Yolov1PredictionHead()

        raise NotImplementedError

    def forward(self, inputs: List[torch.Tensor]) -> List[Dict[torch.Tensor]]:
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
        # if batch size 1,
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.stack(inputs)

        outputs = self.prediction_head(inputs)
        # print(outputs.size())
        if self.training:
            raise NotImplementedError
        else:
            sys.exit(0)

            grid_x = torch.arange(7, device='cuda').repeat(7, 1).view(1, 1, 7, 7)
            grid_y = torch.arange(7, device='cuda').repeat(7, 1).t().view(1, 1, 7, 7)

            norm_x = grid_x * (1./7)
            norm_y = grid_y * (1./7)     

            preds = []
            labels = outputs[...,5*2:]  # torch.Size([B, 7, 7, 20]) -> [[[...num_classes]], [[...]]]
            labels_proba = outputs[...,5*2:].argmax(-1)  # torch.Size([2, 7, 7])
            for i in range(0, 5*2, 5):
                boxes = outputs[...,i:i+4]  # torch.Size([2, 7, 7, 4]) -> [[[x,y,x,y]], [[...]]]
                scores = outputs[...,i+5]  # torch.Size([2, 7, 7]) -> [[[]], [[]]]

                mask = (scores * labels_proba) > 0.5 # torch.Size([2, 7, 7])
                if not mask.size(0):
                    continue
            
                boxes[...,0] = boxes[...,0] * (1./7) + norm_x
                boxes[...,1] = boxes[...,1] * (1./7) + norm_y

                xy_normal = boxes[...,:2]
                wh_normal = boxes[...,2:]

                boxes[...,:2] = xy_normal - 0.5 * wh_normal
                boxes[...,2:] = xy_normal + 0.5 * wh_normal
                preds.append({
                    'boxes': boxes[mask],
                    'scores': scores[mask],
                    'labels': labels[mask]
                })

            # print(preds)
            # preds: torch.Size([num_boxes, 4]), torch.Size([num_boxes, 1]), torch.Size([num_boxes, 20])        
            # NMS 후 selected boxes에 담아서 원본 이미지 크기로 다시 재변환
            selected_boxes = []
            for pred in preds:
                print(len(pred['boxes']))
                print(non_maximum_supression(pred['boxes'], pred['scores']))
                print(len(non_maximum_supression(pred['boxes'], pred['scores'])))


            ## Recover inputs image size!!!
            boxes_detected, class_names_detected, probs_detected = [], [], []
            for b in range(boxes_normalized.size(0)):
                box_normalized = boxes_normalized[b]
                class_label = class_labels[b]
                prob = probs[b]

                x1, x2 = w * box_normalized[0], w * box_normalized[2] # unnormalize x with image width.
                y1, y2 = h * box_normalized[1], h * box_normalized[3] # unnormalize y with image height.
                boxes_detected.append(((x1, y1), (x2, y2)))

                class_label = int(class_label) # convert from LongTensor to int.
                class_name = self.class_name_list[class_label]
                class_names_detected.append(class_name)

                prob = float(prob) # convert from Tensor to float.
                probs_detected.append(prob)

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




        return outputs

    def _transform(self, outputs):
        pass

        

def non_maximum_supression(boxes, scores, threshold=0.5):
    """Non-maximum supression"""
    x1, y1 = boxes[...,0], boxes[...,1]
    x2, y2 = boxes[...,2], boxes[...,3]
    areas = (x2 - x1) * (y2 - y1)
    
    _, indices = scores.sort(0, descending=True)
    
    keep = []
    while indices.numel():
        i = indices.item() if (indices.numel() == 1) else indices[0].item()
        keep.append(i)
        
        if indices.numel() == 1:
            break
        
        # print(x1[i])
        
        inter_x1 = x1[indices[1:]].clamp(min=x1[i].item()) # [m-1, ]
        inter_y1 = y1[indices[1:]].clamp(min=y1[i].item()) # [m-1, ]
        inter_x2 = x2[indices[1:]].clamp(max=x2[i].item()) # [m-1, ]
        inter_y2 = y2[indices[1:]].clamp(max=y2[i].item()) # [m-1, ]

        inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
        inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]

        inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
        unions = areas[i] + areas[indices[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
        ious = inters / unions # [m-1, ]

        ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
        if ids_keep.numel() == 0:
            break # If no box left, break.
        indices = indices[ids_keep+1]
    
    return keep


