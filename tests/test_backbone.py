import torch  
import torch.nn as nn      
import torch.nn.functional as F       
from torch.autograd import Variable


# class ConvLayer(nn.Module):
#     def __init__(self,in_planes,out_planes,kernel_size = 3):
#         super(ConvLayer, self).__init__()
#         padding = kernel_size//2 if kernel_size==3 else 0
#         self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,padding=padding,bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.LeakyReLU(0.1)

#     def forward(self,x):
#         out = self.relu(self.bn(self.conv(x)))
#         return out

# class DarkNet19(nn.Module):
#     def __init__(self,cfg1, cfg2):
#         super().__init__()
#         self.layer1 = self._make_layer(cfg1, in_planes=3)
#         self.layer2 = self._make_layer(cfg2, in_planes=512)

#     def _make_layer(self,cfg,in_planes):
#         layers = []
#         for i in cfg:
#             if i=='M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
#             elif isinstance(i,tuple):
#                 out_planes = i[0]
#                 layers += [ConvLayer(in_planes,out_planes,kernel_size=1)]
#                 in_planes = out_planes
#             else:
#                 layers += [ConvLayer(in_planes,i)]
#                 in_planes = i

#         return nn.Sequential(*layers)

#     def forward(self,x):
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         return x1,x2



# l1 = [32,'M',64,'M',128,(64,1),128,'M',256,(128,1),256,'M',512,(256,1),512,(256,1),512]
# l2 = ['M',1024,(512,1),1024,(512,1),1024]

# from typing import Tuple, List, Dict
# import torch
# from torch import nn, Tensor
# import torch.nn.functional as F


# class Darknet19(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.in_channels = 3
#         self.channels = []
#         self.layers = nn.ModuleList()

#         for cfg in config:
#             self._make_layers(cfg)

#     def forward(self, x):
#         outputs = []
#         for layer in self.layers:
#             x = layer(x)
#             outputs.append(x)
        
#         return x

#     def _make_layers(self, config):
#         layers = []
#         for v in config:
#             kwargs = None
#             if isinstance(v, tuple):
#                 kwargs = v[1]
#                 v = v[0]
            
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 if kwargs is None:
#                     kwargs = {'kernel_size': 3, 'padding': 1}
                
#                 layers += [
#                     nn.Conv2d(
#                         self.in_channels, 
#                         v, 
#                         **kwargs, 
#                         bias=False),
#                 nn.BatchNorm2d(v),
#                 nn.LeakyReLU(0.1)]
    
#                 self.in_channels = v
        
#         self.channels.append(self.in_channels)
#         self.layers.append(nn.Sequential(*layers))

        

# def darknet19(config: Dict = None):
#     model = Darknet19(config)

#     # if isinstance(config, dict):
#     #     if config.backbone.pretrained:
#     #         model.load_state_dict(torch.load(config.backbone.path))

#     return model


from torchsummary import summary
from boda.backbone_darknet import darknet, darknet19
# backbone_layers = [
#     [32, 'M'],
#     [64, 'M'],
#     [128, (64, {'kernel_size': 1}), 128, 'M'],
#     [256, (128, {'kernel_size': 1}), 256, 'M'],
#     [512, (256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512, 'M'],
#     [1024, (512, {'kernel_size': 1}), 1024, (512, {'kernel_size': 1}), 1024]]

# backbone_layers = [
#     [32],
#     ['M', 64],
#     ['M', 128, (64, {'kernel_size': 1}), 128],
#     ['M', 256, (128, {'kernel_size': 1}), 256],
#     ['M', 512, (256, {'kernel_size': 1}), 512, (256, {'kernel_size': 1}), 512],
#     ['M', 1024, (512, {'kernel_size': 1}), 1024, (512, {'kernel_size': 1}), 1024]]

# # model = DarkNet19(l1, l2)
model = darknet()
# model = darknet19(bn=True, relu=True)
print(summary(model, (3, 224, 224), verbose=0))