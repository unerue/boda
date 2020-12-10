import sys
from typing import Tuple, List, Dict, Any
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, Tensor

from ..architecture_base import Neck, Head, Model
from .backbone_darknet import darknet, darknet21
from .configuration_yolov1 import Yolov1Config


class Yolov1PredictNeck(Neck):
    """Prediction Neck for YOLOv1
    Arguments:
        in_channels (int):
    """
    def __init__(self, config, in_channels: int, **kwargs) -> None:
        """
        """
        super().__init__()
        self.config = config
        self.out_channels = []

        self._in_channels = in_channels
        self.layers = nn.ModuleList()

        self._make_layer(1024, config.bn, config.relu)
        self._make_layer(1024, config.bn, config.relu)
        self._make_layer(1024, config.bn, config.relu)
        self._make_layer(1024, config.bn, config.relu)

    def _make_layer(self, out_channels, bn: bool = False, relu: bool = False, **kwargs):
        _layers = []
        _layers.append(
            nn.Conv2d(self._in_channels, out_channels, kernel_size=3, padding=1, **kwargs))

        if bn:
            _layers.append(nn.BatchNorm2d(out_channels))

        if relu:
            _layers.append(nn.ReLU())
        else:
            _layers.append(nn.LeakyReLU(0.1))

        self._in_channels = out_channels
        self.layers.append(nn.Sequential(*_layers))
        self.out_channels.append(out_channels)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        """
            inputs (List[Tensor]): 

        Return:
            (Tensor): Size([])
        """
        inputs = inputs[self.config.selected_layers[0]]
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs


class Yolov1PredictHead(Head):
    """Prediction Neck for YOLOv1
    Arguments:
        selected_layers (List[float]):
        scales (List[float]):
    """
    def __init__(self, config, in_channels: int = 1024, relu: bool = False, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.out_channels = []

        self._out_channels = 5 * config.num_boxes + config.num_classes
        self.layers = nn.Sequential(
            nn.Linear(config.num_grids * config.num_grids * in_channels, 4096),
            nn.LeakyReLU(0.1) if not relu else nn.ReLU(),
            nn.Linear(4096, config.num_grids * config.num_grids * self._out_channels),
            nn.Sigmoid())

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """
        Argument:
            inputs (Tensor): Size([])
        Return:
            Dict[str, Tensor]:
                boxes: Size([batch_size, num_boxes, 4])
                scores: Size([batch_size, num_boxes])
                labels: Size([batch_size, num_boxes, 20])
        """
        bs = inputs.size(0)
        inputs = inputs.view(bs, -1)
        outputs = self.layers(inputs)
        outputs = outputs.view(-1, self.config.num_grids, self.config.num_grids, self._out_channels)

        outputs = outputs.view(bs, -1, 5*self.config.num_boxes+self.config.num_classes)

        boxes = outputs[..., :5*self.config.num_boxes].contiguous().view(bs, -1, 5)
        scores = boxes[..., 4]
        boxes = boxes[..., :4]
        labels = outputs[..., 5*self.config.num_boxes:]
        labels = labels.repeat(1, 2, 1)

        preds = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels}

        return preds


class Yolov1Pretrained(Model):
    config_class = Yolov1Config
    base_model_prefix = 'yolov1'

    # def __init__(self, config):
    #     super().__init__()
        
    # def _init_weights(self, module):
    #     """ Initialize the weights """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()


class Yolov1Model(Yolov1Pretrained):
    """
    ██╗   ██╗ ██████╗ ██╗      ██████╗           ████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗          ╚═██║
     ╚████╔╝ ██║   ██║██║     ██║   ██║██╗   ██╗   ██║
      ╚██╔╝  ██║   ██║██║     ██║   ██║╚██╗ ██╔╝   ██║
       ██║   ╚██████╔╝███████╗╚██████╔╝ ╚████╔╝  ██████╗
       ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝   ╚═══╝   ╚═════╝

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
    def __init__(self, config, backbone=None, neck=None, head=None, **kwargs) -> None:
        super().__init__(config)
        self.config = config
        if backbone is None:
            self.backbone = darknet(pretrained=False)
            # self.backbone = darknet21()
        if neck is None:
            self.neck = Yolov1PredictNeck(config, self.backbone.out_channels[-1])
        if head is None:
            self.head = Yolov1PredictHead(config, self.neck.out_channels[-1])

        # self._init_weights()

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        Argument:
            inputs (List(FloatTensor[C, H, W]): Number of batch size Size([B, C, H, W]))
        Return:
            outputs
        """
        inputs = self.check_inputs(inputs)
        self.config.device = inputs.device
        self.config.batch_size = inputs.size(0)

        if self.training:
            outputs = self.backbone(inputs)
            outputs = self.neck(outputs)
            outputs = self.head(outputs)

            return outputs
        else:
            return inputs
