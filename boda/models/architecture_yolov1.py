import os
from typing import Tuple, List, Dict, Any, Union
from collections import OrderedDict

import torch
from torch import nn, Tensor

from ..architecture_base import Neck, Head, Model
from ..backbones.darknet import darknet
from .configuration_yolov1 import Yolov1Config


class Yolov1PredictNeck(Neck):
    """Prediction Neck for YOLOv1

    Arguments:
        in_channels (int):

    Keyword Arguments:
        bn (bool)
        relu (bool)
    """
    def __init__(self, config, in_channels: int = 1024, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.channels = []
        self.selected_layers = config.selected_layers
        if isinstance(self.selected_layers, list):
            self.selected_layers = self.selected_layers[0] 

        self._in_channels = in_channels

        self.layers = nn.ModuleList()
        self._make_layer(in_channels)
        self._make_layer(in_channels)
        self._make_layer(in_channels)
        self._make_layer(in_channels)

    def _make_layer(
        self,
        out_channels,
        bn: bool = False,
        relu: bool = False,
        **kwargs
    ) -> None:
        _layers = []
        _layers.append(
            nn.Conv2d(
                self._in_channels, out_channels,
                kernel_size=3, padding=1, **kwargs))

        if bn:
            _layers.append(nn.BatchNorm2d(out_channels))

        if relu:
            _layers.append(nn.ReLU())
        else:
            _layers.append(nn.LeakyReLU(0.1))

        self._in_channels = out_channels
        self.layers.append(nn.Sequential(*_layers))
        self.channels.append(out_channels)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        """
        Arguments:
            inputs (List[Tensor]):

        Return:
            (Tensor): Size([])
        """
        inputs = inputs[self.selected_layers]
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs


class Yolov1PredictHead(Head):
    """Prediction Neck for YOLOv1

    Arguments:
        config
        in_channles (int):
        out_channels (int):
        relu (bool):
    """
    def __init__(
        self,
        config,
        in_channels: int = 1024,
        out_channels: int = 4006,
        relu: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.config = config
        self.channels = []  # TODO: out_channels? channels?

        self._out_channels = 5 * config.num_boxes + config.num_classes
        self.layers = nn.Sequential(
            nn.Linear(
                config.num_grids * config.num_grids * in_channels,
                out_channels),
            nn.LeakyReLU(0.1) if not relu else nn.ReLU(),
            nn.Linear(
                out_channels,
                config.num_grids * config.num_grids * self._out_channels),
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
        outputs = outputs.view(
            -1, self.config.num_grids, self.config.num_grids, self._out_channels)

        outputs = outputs.view(
            bs, -1, 5*self.config.num_boxes+self.config.num_classes)

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

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = Yolov1Model(config)
        # model.state_dict(torch.load('yolact.pth'))
        return model

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


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
                in [xmin, ymin, xmax, ymax] format, ranging from 0 to W and 0 to H
            labels (Int64Tensor[N]): the label for each bounding box. 0 represents 
                always the background class.
            image_id (Int64Tensor[1]): an image identifier. It should be unique 
                between all the images in the dataset, and is used during evaluation
            area (Tensor[N]): The area of the bounding box. This is used during 
                evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
            iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored 
                during evaluation.
    """
    def __init__(
        self,
        config,
        backbone=None,
        neck=None,
        head=None,
        **kwargs
    ) -> None:
        super().__init__(config)
        self.config = config
        if backbone is None:
            self.backbone = darknet(pretrained=False)
        if neck is None:
            self.neck = Yolov1PredictNeck(config, self.backbone.channels[-1])
        if head is None:
            self.head = Yolov1PredictHead(config, self.neck.channels[-1])


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
