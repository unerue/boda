from typing import Tuple, List, Dict, Any, Callable, TypeVar
from collections import defaultdict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import os
import math
import itertools

from .architecture_base import BaseModel
from .backbone_resnet import resnet101
from ..base import Neck, Head, Model

# ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
# script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

class Yolov3PredictNeck(nn.Module):
    def __init__(self, config, in_channels) -> None:
        super().__init__()
        self.config = config
        

class Yolov3PredictHead(nn.Module):
    def __init__(self, config, in_channels, out_channels, aspect_ratios, scales, parent, index) -> None:
        super().__init__()
        self.config = config
        

class Yolov3Model(BaseModel):
    def __init__(self, config, backbone=None, neck=None):
        super().__init__()
        self.config = config
       