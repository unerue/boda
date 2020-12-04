import sys
from typing import Tuple, List, Dict, Any
import numpy as np
from torch import nn, Tensor


class PriorBox(nn.Module):
    def __init__(self) -> None:
        raise NotImplementedError


class Yolov4PredictNeck(nn.Module):
    """Prediction Neck for Yolov4
    Arguments:
        selected_layers (List[float]):
        scales (List[float]):
    Returns:
        List[Tensor]
    """
    def __init__(self, **kwargs) -> None:
        self.selected_layers = kwargs.get('selected_layers')
        self.scales = kwargs.get('scales')
        raise NotImplementedError
    
    def forward(self, inputs) -> List[Tensor]:
        outputs = inputs
        return outputs

class Yolov4PredictHead(nn.Module):
    def __init__(self) -> None:
        raise NotImplementedError

    def forward(self, inputs) -> List[Tensor]:
        outputs = inputs
        return outputs


class Yolov4Model(nn.Module):
    """
    ██╗   ██╗ ██████╗ ██╗      ██████╗ 
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗
     ╚████╔╝ ██║   ██║██║     ██║   ██║██╗   ██╗
      ╚██╔╝  ██║   ██║██║     ██║   ██║╚██╗ ██╔╝ 
       ██║   ╚██████╔╝███████╗╚██████╔╝ ╚████╔╝    
       ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝    ╚══╝

    The only specificity that we require is that the dataset 
    __getitem__ should return:

    Arguments:
        image: a PIL Image of size (H, W)
        target: a dict containing the following fields
            boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes 
                in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            labels (Int64Tensor[N]): the label for each bounding box. 0 represents 
                always the background class.
            image_id (Int64Tensor[1]): an image identifier. It should be unique 
                between all the images in the dataset, and is used during evaluation
            area (Tensor[N]): The area of the bounding box. This is used during 
                evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
            iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored 
                during evaluation.
    """
    def __init__(self) -> None:
        raise NotImplementedError

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        outputs = inputs
        return outputs

    def initialize_weights(self, path):
        raise NotImplementedError


    