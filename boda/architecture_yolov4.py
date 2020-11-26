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
    """
    def __init__(self) -> None:
        raise NotImplementedError

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        outputs = inputs
        return outputs

    def initialize_weights(self, path):
        raise NotImplementedError
    
    