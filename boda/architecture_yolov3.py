import sys
from typing import Tuple, List, Dict, Any
import numpy as np
from torch import nn, Tensor


class PriorBox(nn.Module):
    def __init__(self) -> None:
        raise NotImplementedError


class Yolov3PredictNeck(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.selected_layers = kwargs.get('selected_layers')
        self.scales = kwargs.get('scales')
        self.
        raise NotImplementedError


class Yolov3PredictHead(nn.Module):
    def __init__(self) -> None:
        raise NotImplementedError



class Yolov3Model(nn.Module):
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

    def forward(self) -> List[Tensor]:
        raise NotImplementedError
    