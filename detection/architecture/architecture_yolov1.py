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
        self.grid_size =7
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
        pass

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.stack(inputs)

        print(inputs.size())
        
        outputs = self.prediction_head(inputs)
        print(outputs.size())

        # outputs = outputs.view(outputs.size(0), 1, 30, outputs.size(2), outputs.size(2)).permute(0, 1, 3, 4, 2).contiguous()
        print(outputs[0][0].size(), outputs[0][0][0])
        outputs = outputs.view(outputs.size(0), 7*7, 30)
        print(outputs.size())
        print(outputs[0][0])
        print(outputs[1][0])

        boxes = outputs[...,:2*5]
        labels = outputs[...,2*5:]
        print('before:', boxes[0][0])
        print('before:', boxes[1][0])
        print(boxes.size(), labels.size())
        boxes = boxes.view(-1,5*2)
        labels = labels.view(-1,20)
        print(boxes.size(), labels.size())
        print('after:', boxes[0])
        print('after:', boxes[49])

        cell_size = 1./14
        print(len(boxes))
        for i in range(len(boxes)):
            for j in range(0, 2*5, 5): # num_bboxes B * 5) + C
                bbox = boxes[i,j:j+5]
                x = bbox[0].item()
                y = bbox[1].item()
                w = bbox[2]
                h = bbox[3]
                score = bbox[5]
                print(bbox.size(), bbox)
                # boxes[i,:2] = , boxes[i,:2] * cell_size
                # print(boxes)

        return outputs

    def _transform(self, outputs):
        pass

        




def non_maximum_supression(boxes, scores, threshold=0.5):
    pass



