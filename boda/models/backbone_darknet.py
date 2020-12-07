import math
from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..base import Backbone

DARKNET_PRETRAINED_CONFIG_MAP = {
    'darknet-base': None,
    'darknet19-base': None,
    'darknet21': None,
    'darknet53': None,
}

DARKNET_STRUCTURES = {
    'darknet-base': [
        [(64, {'kernel_size': 7, 'padding': 2}), 'M'],
        [192, 'M'],
        [(128, {'kernel_size': 1}), 256, (256, {'kernel_size': 1}), 512, 'M'],
        [
            (256, {'kernel_size': 1}), 512, 
            (256, {'kernel_size': 1}), 512, 
            (256, {'kernel_size': 1}), 512, 
            (256, {'kernel_size': 1}), 512,
            (512, {'kernel_size': 1}), 1024, 'M'],
        [
            (512, {'kernel_size': 1}), 1024, 
            (512, {'kernel_size': 1}), 1024, 1024, 1024]],
    'darknet-tiny': [],
    'darknet19-base': [
        [32],
        ['M', 64],
        ['M', 128, (64, {'kernel_size': 1}), 128],
        ['M', 256, (128, {'kernel_size': 1}), 256],
        ['M', 512, (256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512],
        ['M', 1024, (512, {'kernel_size': 1}), 1024, (512, {'kernel_size': 1}), 1024]],
    'darknet19-tiny': [],
}


class DarkNet19(nn.Module):
    """DarkNet19 for YOLOv1, v2 backbone
    """
    def __init__(
        self, backbone_structure: List, bn: bool = False, relu: bool = False):
        super().__init__()
        self.bn = bn
        self.relu = relu

        self.in_channels = 3
        self.channels = []
        self.layers = nn.ModuleList()

        for structure in backbone_structure:
            self._make_layers(structure)

    def forward(self, inputs: Tensor) -> List[Tensor]:
        outputs = []
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)
        
        return outputs

    def _make_layers(self, layers: List):
        _layers = []
        for v in layers:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]
            
            if v == 'M':
                _layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if kwargs is None:
                    kwargs = {'kernel_size': 3, 'padding': 1}
                
                _layers += [
                    nn.Conv2d(
                        in_channels=self.in_channels, 
                        out_channels=v, 
                        bias=False,
                        **kwargs)]
                if self.bn:
                    _layers += [
                        nn.BatchNorm2d(v),
                        nn.LeakyReLU(0.1) if not self.relu else nn.ReLU()]
                else:
                    _layers += [
                        nn.LeakyReLU(0.1) if not self.relu else nn.ReLU()]
    
                self.in_channels = v
        
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*_layers))


class Shortcut(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: List[int], residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels[0], 
            kernel_size=1,
            stride=1, 
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels[0])
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels[0], 
            out_channels=out_channels[1], 
            kernel_size=3,
            stride=1, 
            padding=1, 
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, inputs: Tensor):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.residual:
            out += residual
    
        return out


class DarkNet(nn.Module):
    """
    References:
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        https://arxiv.org/abs/1804.02767
    """
    def __init__(self, layers: List[int], block=Shortcut):
        super().__init__()
        self.in_channels = 32
        self.conv = nn.Conv2d(
            in_channels=3, 
            out_channels=self.in_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.LeakyReLU(0.1)

        self.layers = nn.ModuleList()
        self.channels = []

        self._make_layer(block, [32, 64], layers[0])
        self._make_layer(block, [64, 128], layers[1])
        self._make_layer(block, [128, 256], layers[2])
        self._make_layer(block, [256, 512], layers[3])
        self._make_layer(block, [512, 1024], layers[4])
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channels: List[int], blocks: List[int]):
        layers = []
        layers.append(nn.Conv2d(
            self.in_channels, 
            out_channels[1], 
            kernel_size=3,
            stride=2, 
            padding=1, 
            bias=False))
        layers.append(nn.BatchNorm2d(out_channels[1]))
        layers.append(nn.LeakyReLU(0.1))
        
        self.in_channels = out_channels[1]
        for _ in range(0, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        self.layers.append(nn.Sequential(*layers))
        self.channels.append(out_channels[1])

    def forward(self, inputs: Tensor) -> List[Tensor]:
        inputs = self.conv(inputs)
        inputs = self.bn(inputs)
        inputs = self.relu(inputs)

        outputs = []
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)
             
        return outputs

    def initialize_weights(self, path):
        state_dict = torch.load(path)
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx-1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)
        
        self.load_state_dict(state_dict, strict=False)


def darknet(structure_or_name: str = 'darknet-base', pretrained: bool = False, **kwargs):
    if isinstance(structure_or_name, str):
        backbone = DarkNet19(DARKNET_STRUCTURES[structure_or_name], **kwargs)

    return backbone


def darknet19(structure_or_name: str = 'darknet19-base', pretrained: bool = False, **kwargs):
    if isinstance(structure_or_name, str):
        backbone = DarkNet19(DARKNET_STRUCTURES[structure_or_name], **kwargs)
        
    return backbone


def darknet21(pretrained=False, **kwargs):
    backbone = DarkNet([1, 1, 2, 2, 1])
    if pretrained:
        backbone.load_state_dict(torch.load(pretrained))

    return backbone


def darknet53(pretrained=False, **kwargs):
    backbone = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        backbone.load_state_dict(torch.load(pretrained))
        
    return backbone





# class Yolov3Backbone:
#     def __init__(self):
#         raise NotImplementedError


# def parse_model_config(path):
#     """Parses the yolo-v3 layer configuration file and returns module definitions"""
#     with open(path, 'r') as file:
#         lines = file.read().split('\n')
#         lines = [x for x in lines if x and not x.startswith('#')]
#         lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
#         lines = lines[:lines.index('[backbone]')]

#     modules = []
#     for line in lines:
#         if line.startswith('['): # This marks the start of a new block
#             modules.append({})
#             modules[-1]['type'] = line[1:-1].rstrip()
#             if modules[-1]['type'] == 'convolutional':
#                 modules[-1]['batch_normalize'] = 0
#             if modules[-1]['type'] == 'backbone':
#                 break

#             # if modules[-1]['type'] == 'END':
#             #     break
#         else:
#             key, value = line.split('=')
#             value = value.strip()
#             modules[-1][key.rstrip()] = value.strip()

#     return modules


# class Upsample(nn.Module):
#     def __init__(self, scale_factor, mode='nearest'):
#         super().__init__()
#         self.scale_factor = scale_factor
#         self.mode = mode

#     def forward(self, x):
#         return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


# class EmptyLayer(nn.Module):
#     def __init__(self):
#         super().__init__()



# class Darknet(nn.Module):
#     def __init__(self, config_path, img_size=416):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.channels = []

#         self.config = parse_model_config(config_path)
#         self._make_layers(self.config)

#         self.backbone_modules = [m for m in self.modules() if isinstance(m, (nn.Conv2d, EmptyLayer))]

#     def forward(self, x):
#         outs = []
#         for config, layer in zip(self.config, self.layers):
#             if config['type'] in ['convolutional', 'maxpool']:
#                 x = layer(x)
        
#             elif config['type'] == 'shortcut':
#                 x = outs[-1] + outs[int(config['from'])]

#             outs.append(x)

#         return x

#     def _make_layers(self, config):
#         self.channels.append(int(config.pop(0)['channels']))
#         for module in config:
#             modules = []
#             if module['type'] == 'convolutional':
#                 bn = int(module['batch_normalize'])
#                 bias = True if not bn else False
#                 out_channels = int(module['filters'])
#                 kernel_size = int(module['size'])
#                 padding = (kernel_size - 1) // 2
#                 modules += [nn.Conv2d(
#                         in_channels=self.channels[-1],
#                         out_channels=out_channels,
#                         kernel_size=kernel_size,
#                         stride=int(module['stride']),
#                         padding=padding,
#                         bias=bias)]
#                 if bn:
#                     modules += [nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)]
                        
#                 if module['activation'] == 'leaky':
#                     modules += [nn.LeakyReLU(0.1)]

#             elif module['type'] == 'shortcut':
#                 out_channels = self.channels[1:][int(module['from'])]
#                 modules += [EmptyLayer()]

#             self.channels.append(out_channels)
#             self.layers.append(nn.Sequential(*modules))


# if __name__ == '__main__':
#     from torchsummary import summary

#     config_path = '/home/unerue/Documents/computer-vision/detection/backbone/yolov3.cfg'
#     backbone = Darknet(config_path)

    
#     print(summary(backbone, input_data=(3, 416, 416), verbose=0))
#     print(len(backbone.backbone_modules))

#     cnt = 0
#     for i in backbone.config:
#         if i['type'] == 'convolutional':
#             cnt += 1
#             print(cnt, 'convolutional')
#         elif i['type'] == 'backbone':
#             print('='*50)
#         elif i['type'] == 'upsample':
#             cnt += 1
#             print(cnt, 'upsample')

#         elif i['type'] == 'route':
#             cnt += 1
#             print(cnt, 'route**********')

#         elif i['type'] == 'shortcut':
#             cnt += 1
#             print(cnt, 'shortcut')

#         elif i['type'] == 'yolo':
#             cnt += 1
#             print(cnt, 'yolo')

#     print(backbone.backbone_modules[61])
#     print(len(backbone.backbone_modules))

#     # pprint.pprint(parse_model_config('/home/unerue/Documents/computer-vision/detection/backbone/yolov3.cfg'))