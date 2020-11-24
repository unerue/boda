import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    with open(path, 'r') as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    
    modules = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            modules.append({})
            modules[-1]['type'] = line[1:-1].rstrip()
            if modules[-1]['type'] == 'convolutional':
                modules[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            value = value.strip()
            modules[-1][key.rstrip()] = value.strip()

    return modules


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()



class TestYolov3(nn.Module):
    def __init__(self, config_path):
        super().__init__()
     
        self.layers = nn.ModuleList()
        self.channels = []

        self.config = parse_model_config(config_path)
        self._make_layers(self.config)

    

    def forward(self, x):
        outs = []
        for config, layer in zip(self.config, self.layers):
            if config['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = layer(x)
                
            elif config['type'] == 'route':
                x = torch.cat([outs[int(i)] for i in config['layers'].split(',')], 1)

            elif config['type'] == 'shortcut':
                x = outs[-1] + outs[int(config['from'])]
            # elif module_def["type"] == "yolo":
            #     x, layer_loss = module[0](x, targets, img_dim)
            #     loss += layer_loss
            #     yolo_outputs.append(x)
            outs.append(x)

        # yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        # return yolo_outputs if targets is None else (loss, yolo_outputs)
        
        return outs

    def _make_layers(self, config):
        self.channels.append(int(config.pop(0)['channels']))
        for module in config:
            modules = []
            if module['type'] == 'convolutional':
                bn = int(module['batch_normalize'])
                bias = True if not bn else False
                out_channels = int(module['filters'])
                kernel_size = int(module['size'])
                padding = (kernel_size - 1) // 2
                modules += [nn.Conv2d(
                        in_channels=self.channels[-1],
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=int(module['stride']),
                        padding=padding,
                        bias=bias)]
                if bn:
                    modules += [nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)]
                        
                if module['activation'] == 'leaky':
                    modules += [nn.LeakyReLU(0.1)]

            elif module['type'] == 'maxpool':
                kernel_size = int(module['size'])
                stride = int(module['stride'])
                if kernel_size == 2 and stride == 1:
                    modules += [nn.ZeroPad2d((0, 1, 0, 1))]
                modules += [nn.MaxPool2d(
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=int((kernel_size-1) // 2))]

            elif module['type'] == 'upsample':
                modules += [Upsample(scale_factor=int(module['stride']), mode='nearest')]

            elif module['type'] == 'route':
                layers = [int(x) for x in module['layers'].split(',')]
                out_channels = sum([self.channels[1:][i] for i in layers])
                modules += [EmptyLayer()]

            elif module['type'] == 'shortcut':
                out_channels = self.channels[1:][int(module['from'])]
                modules += [EmptyLayer()]

            self.channels.append(out_channels)
            self.layers.append(nn.Sequential(*modules))
