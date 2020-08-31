
from typing import Tuple, List, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Darknet19(nn.Module):
    def __init__(self, config, num_classes=1000):
        super().__init__()
        self.in_channels = 3
        self.channels = []
        self.layers = nn.ModuleList()

        for cfg in config:
            self._make_layers(cfg)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        
        return x

    def _make_layers(self, config):
        layers = []
        for v in config:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]
            
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if kwargs is None:
                    kwargs = {'kernel_size': 3, 'padding': 1}
                print(v, kwargs)
                layers += [
                    nn.Conv2d(
                        self.in_channels, 
                        v, 
                        **kwargs, 
                        bias=False),
                nn.BatchNorm2d(v),
                nn.LeakyReLU(0.1)]
    
                self.in_channels = v
        
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))

        

def darknet19(config: Dict = None):
    model = Darknet19(config)

    # if isinstance(config, dict):
    #     if config.backbone.pretrained:
    #         model.load_state_dict(torch.load(config.backbone.path))

    return model