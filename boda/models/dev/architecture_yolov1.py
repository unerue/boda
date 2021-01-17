import os
from typing import Tuple, List, Dict, Any, Union, Optional, Callable
from collections import OrderedDict

import torch
from torch import nn, Tensor

from ..architecture_base import Neck, Head, Model
from ..modules import Conv2dDynamicSamePadding
from .backbone_darknet import darknet
from .backbone_resnet import resnet50, resnet34, resnet18
from .configuration_yolov1 import Yolov1Config


class Yolov1PredictNeck(Neck):
    """Prediction Neck for YOLOv1

    Args:
        in_channels (int):
        bn (bool):
        relu (bool):
    """
    def __init__(
        self,
        config,
        channels: List[int],
        in_channels: int = 1024,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        **kwargs
    ) -> None:
        super().__init__()
        self.config = config
        self.selected_layers = kwargs.get('select_layers')
        self.norm_layer = norm_layer

        if self.config is not None:
            for k, v in config.to_dict().items():
                setattr(self, k, v)

        self.channels = []
        self._in_channels = in_channels

        self.layers = nn.ModuleList()
        self._make_layer(channels[-1])
        self._make_layer(in_channels)
        # self._make_layer(in_channels)
        # self._make_layer(in_channels)

    def _make_layer(
        self,
        out_channels,
        **kwargs
    ) -> None:
        _layers = []
        _layers.append(
            # nn.Conv2d(
            #     self._in_channels, out_channels,
            #     kernel_size=3, padding=1, **kwargs))
            Conv2dDynamicSamePadding(
                self._in_channels, out_channels,
                kernel_size=3, **kwargs))

        if self.norm_layer is not None:
            _layers.append(self.norm_layer(out_channels))

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
        inputs = inputs[-1]
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs


class Yolov1PredictHead(Head):
    """Prediction Neck for YOLOv1

    Args:
        config
        in_channles (int):
        out_channels (int):
        relu (bool):
    """
    def __init__(
        self,
        config,
        in_channels: int = 1024,
        out_channels: int = 4096,
        **kwargs
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = kwargs.get('num_classes')
        self.num_grids = kwargs.get('num_grids')
        self.num_boxes = kwargs.get('num_boxes')

        if config is not None:
            for k, v in config.to_dict().items():
                setattr(self, k, v)

        self.channels = []
        self._out_channels = 5 * config.num_boxes + config.num_classes
        self.layers = nn.Sequential(
            # nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            # nn.Linear(
            #     config.num_grids * config.num_grids * in_channels,
            #     out_channels),
            nn.Linear(100352, out_channels), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(
                out_channels,
                config.num_grids * config.num_grids * self._out_channels),
            nn.Sigmoid()
        )

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            inputs (Tensor): Size([])

        Returns:
            Dict[str, Tensor]:
                boxes: Size([batch_size, num_boxes, 4])
                scores: Size([batch_size, num_boxes])
                labels: Size([batch_size, num_boxes, 20])
        """
        batch_size = inputs.size(0)
        # print(inputs.size())
        # inputs = nn.AdaptiveAvgPool2d((7, 7))(inputs)
        # inputs = inputs.view(batch_size, -1)
        # print('avg', inputs.size())
        # inputs = nn.Flatten()(inputs)
        # print(inputs.size())
        outputs = self.layers(inputs)
        outputs = outputs.view(
            -1, self.num_grids, self.num_grids, self._out_channels)

        # outputs = outputs.view(
        #     batch_size, -1, 5*self.num_boxes+self.num_classes)

        # boxes = outputs[..., :5*self.num_boxes].contiguous().view(batch_size, -1, 5)
        # scores = boxes[..., 4]
        # boxes = boxes[..., :4]
        # labels = outputs[..., 5*self.num_boxes:]
        # labels = labels.repeat(1, 2, 1)

        # preds = {
        #     'boxes': boxes,
        #     'scores': scores,
        #     'labels': labels}
        # outputs = outputs.clamp(min=0)

        return outputs


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

    Args:
        images: a PIL Image of size (H, W)
        targets: a dict containing the following fields
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
        self.num_boxes = config.num_boxes
        self.num_classes = config.num_classes
        self.score_threshold = kwargs.get('score_threshold', 0.2)

        if backbone is None:
            # self.backbone = darknet(pretrained=False)
            self.backbone = resnet34()
            # self.backbone.from_pretrained('cache/resnet50-19c8e357.pth')
            self.backbone.from_pretrained('cache/resnet34-333f7ec4.pth')
            # self.backbone.from_pretrained('cache/resnet18-5c106cde.pth')
        if neck is None:
            self.neck = Yolov1PredictNeck(config, self.backbone.channels, self.backbone.channels[-1])
        if head is None:
            self.head = Yolov1PredictHead(config, self.neck.channels[-1])
            # self.head = Yolov1PredictHead(config, self.backbone.channels[-1])

        # self.init_weights()

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
        batch_size = inputs.size(0)

        if self.training:
            outputs = self.backbone(inputs)
            # for out in outputs:
            #     print(out.size())
            # outputs = self.neck(outputs)
            outputs = self.head(outputs[-1])
            # outputs = self.head(outputs)

            # outputs = outputs.view(
                # batch_size, -1, 5*self.num_boxes+self.num_classes)

            # boxes = outputs[..., :5*self.num_boxes].contiguous().view(batch_size, -1, 5)
            # scores = boxes[..., 4]
            # boxes = boxes[..., :4]

            # labels = outputs[..., 5*self.num_boxes:]
            # # labels = labels.view(-1, self.num_classes)
            # # labels = labels.contiguous().view(-1, self.num_classes)
            # # repeat = torch.LongTensor([self.num_boxes]).repeat(labels.size(0)).to(inputs.device)
            # # labels = torch.repeat_interleave(labels, repeat, dim=0)
            # # labels = labels.view(batch_size, boxes.size(1), self.num_classes)

            # return_dict = {
            #     'boxes': boxes,
            #     'scores': scores,
            #     'labels': labels
            # }

            # return return_dict
            return outputs
        else:
            outputs = self.backbone(inputs)
            # outputs = self.neck(outputs)
            # print(outputs.size())
            # TODO: return gpu? or cpu?
            # outputs = self.head(outputs).detach().cpu()
            outputs = self.head(outputs[-1])

            return outputs

            # cell_size = 1.0 / self.num_grids
            # boxes = []
            # scores = []
            # labels = []
            # for i in range(self.num_grids):
            #     for j in range(self.num_grids):
            #         scores, labels = torch.max(outputs[j, i, 5*2:], dim=0)

            #         for b in range(self.num_boxes):
            #             score = outputs[j, i, 5*b + 4]
            #             prob = score * scores
            #             if float(prob) < self.score_threshold:
            #                 continue

            #             # Compute box corner (x1, y1, x2, y2) from tensor.
            #             box = outputs[j, i, 5*b : 5*b + 4]
            #             norm_x0y0 = torch.FloatTensor([i, j]) * cell_size
            #             norm_xy = box[:2] * cell_size + norm_x0y0
            #             norm_wh = box[2:]

            #             box = torch.FloatTensor(4)
            #             box[:2] = norm_xy - 0.5 * norm_wh
            #             box[2:] = norm_xy + 0.5 * norm_wh

            #             boxes.append(box)
            #             scores.append(score)
            #             labels.append(scores)

            # if len(boxes) > 0:
            #     return {
            #         'boxes': torch.stack(boxes, dim=0),
            #         'scores': torch.stack(scores, dim=0),
            #         'labels': torch.stack(labels, dim=0)
            #     }
            # else:
            #     return {
            #         'boxes': torch.FloatTensor(0, 4),
            #         'scores': torch.FloatTensor(0),
            #         'labels': torch.FloatTensor(0)
            #     }

    def init_weights(self):
        # print(self.backbone.backbone_modules)
        for i, (name, module) in enumerate(self.named_modules()):
            if i != 0 and module not in self.backbone.backbone_modules:
                # print('\t', i)
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    nn.init.constant_(module.bias, 0)

        # print(self.backbone.backbone_modules)
