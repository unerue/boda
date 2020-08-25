import os
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import itertools 


from ..backbone import vgg


backbone_layers = [
    [64, 64],
    [ 'M', 128, 128],
    [ 'M', 256, 256, 256],
    [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
    [ 'M', 512, 512, 512],
    # [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
    #  (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
    #  (1024, {'kernel_size': 1})]
]


extra_layers = [
    [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
     (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
     (1024, {'kernel_size': 1})], 
    [(256, {'kernel_size': 1}), 
     (512, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
    [(128, {'kernel_size': 1}), 
     (256, {'kernel_size': 3, 'stride':  2, 'padding':  1})], 
    [(128, {'kernel_size': 1}), 
     (256, {'kernel_size': 3})], 
    [(128, {'kernel_size': 1}), 
     (256, {'kernel_size': 3})]]


# class L2Norm(nn.Module):
#     def __init__(self, n_channels, scale):
#         super().__init__()
#         self.n_channels = n_channels
#         self.gamma = scale or None
#         self.eps = 1e-10
#         self.weight = nn.Parameter(torch.Tensor(self.n_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.constant_(self.weight, val=self.gamma)

#     def forward(self, x):
#         norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
#         #x /= norm
#         x = torch.div(x, norm)
#         out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x

#         return out

config = {
    'selected_layers': [4, 6, 6, 6, 4, 4],
    'num_classes': 20,
    # Prior box configs
    'min_dim': 300,
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],

    'feature_maps': [38, 19, 10, 5, 3, 1], # 'feature_map_sizes'

    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'steps': [8, 16, 32, 64, 100, 300],
    'clip': True,
}


class PriorBox:
    """Prior Box
    Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, config):
        super().__init__()
        self.image_size = config['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(config['aspect_ratios'])
        self.aspect_ratios = config['aspect_ratios']

        self.variance = config['variance'] or [0.1]
        self.feature_maps = config['feature_maps']
        self.min_sizes = config['min_sizes']
        self.max_sizes = config['max_sizes']
        self.steps = config['steps']
        
        self.clip = config['clip']
        self.version = config['name']

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = math.sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                    mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
        # back to torch land
        output = torch.FloatTensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
            
        return output




class SsdPredictionHead(nn.Module):
    """SSD Prediction Neck and Head
    """
    def __init__(self, config = None):
        super().__init__()

        self.num_classes = 20 + 1

        # self.backbone_layers = config.backbone_layers
        self.backbone = vgg(backbone_layers)

        self.extra_channels = []
        self.extra_channels.append(self.backbone.channels[-1])
        self.extra_layers = nn.ModuleList()

        self.in_channels = self.backbone.channels[-1]
        
        for cfg in extra_layers:
            self._add_extra_layers(cfg)

        print(self.extra_channels)
        

        self.selected_layers = [4, 6]

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        
        self._multibox()

        self.l2_norm = None
        self.prior_box = None
        print(self.loc_layers)
        print(self.conf_layers)
        print(len(self.loc_layers))
        print(len(self.conf_layers))

        # self._multibox(config.selected_layers)
        
    def _add_extra_layers(self, config):
        # Extra layers added to VGG for feature scaling
        layers = []
        for v in config:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]
            print(self.in_channels, v, kwargs)
            if v == 'M':
                layers.append(nn.MaxPool2d(**kwargs))
            else:                         
                layers += [
                    nn.Conv2d(
                        in_channels=self.in_channels, 
                        out_channels=v,
                        **kwargs),
                    nn.ReLU()]

                self.in_channels = v

        self.extra_channels.append(v)
        self.extra_layers.append(nn.Sequential(*layers))

    def _multibox(self):
        """config -? VGG selected layers"""
        config = [4, 6, 6, 6, 4, 4]
        for i in range(len(self.extra_channels)):
            self.loc_layers += [
                nn.Conv2d(
                    in_channels=self.extra_channels[i],
                    out_channels=config[i] * 4, 
                    kernel_size=3, 
                    padding=1)]

            self.conf_layers += [
                nn.Conv2d(
                    in_channels=self.extra_channels[i],
                    out_channels=config[i] * self.num_classes, 
                    kernel_size=3, 
                    padding=1)]


    def forward(self, x):
        outputs = []
        x = self.backbone(x)
        # TODO:
        # 마지막 백본 레이어에 L2Norm 추가
        # PriorBox 추가
        x = x[-1]

        outputs.append(x)
        for layer in self.extra_layers:
            x = layer(x)
            outputs.append(x)

        loc_layers = []
        conf_layers = []
        for output, loc_layer, conf_layer in zip(outputs, self.loc_layers, self.conf_layers):
            loc_layers.append(loc_layer(output).permute(0, 2, 3, 1).contiguous())
            conf_layers.append(conf_layer(output).permute(0, 2, 3, 1).contiguous())

        boxes = torch.cat([out.view(out.size(0), -1) for out in loc_layers], dim=1)
        scores = torch.cat([out.view(out.size(0), -1) for out in conf_layers], dim=1)

        outputs = {
            'boxes': boxes.view(boxes.size(0), -1, 4),
            'scores': scores.view(scores.size(0), -1, self.num_classes),
            # 'priors': self.priors
        }
        print(outputs['boxes'].size())
        print(outputs['scores'].size())
        return outputs

#     def _multibox(self, config):
#         """config -? VGG selected layers"""
#         backbone_sources = [21, -2]
#         for k, v in enumerate(backbone_sources):
#             self.loc_layers += [
#                 nn.Conv2d(
#                     self.backbone[v].out_channels,
#                     config[k] * 4, 
#                     kernel_size=3, 
#                     padding=1)]
#             self.conf_layers += [
#                 nn.Conv2d(
#                     self.backbone[v].out_channels,
#                     config[k] * self.num_classes, 
#                     kernel_size=3, 
#                     padding=1)]

#         for k, v in enumerate(self.extra_layers[1::2], 2):
#             self.loc_layers += [
#                 nn.Conv2d(
#                     v.out_channels, 
#                     config[k] * 4, 
#                     kernel_size=3, 
#                     padding=1)]

#             self.conf_layers += [
#                 nn.Conv2d(
#                     v.out_channels, 
#                     config[k] * self.num_classes, 
#                     kernel_size=3, 
#                     padding=1)]

#     def forward(self, inputs):


#         return 


# def _add_extra_layers(self, config):
#         # Extra layers added to VGG for feature scaling
#         flag = False
#         layers = []
#         for k, v in enumerate(config):
#             kwargs = None
#             if isinstance(v, tuple):
#                 kwargs = v[1]
#                 v = v[0]
            
#             if v == 'M':
#                 layers.append(nn.MaxPool2d(**kwargs))
#             else:
                           
#                 if v == 'S':
#                     layer += [
#                         nn.Conv2d(
#                             self.in_channels, 
#                             config[k + 1],
#                             kernel_size=(1, 3)[flag], 
#                             stride=2, 
#                             padding=1)]
#                 else:
#                     layer += [
#                         nn.Conv2d(
#                             self.in_channels, 
#                             v, 
#                             kernel_size=(1, 3)[flag])]
#                 flag = not flag

#             self.in_channels = v

#         self.extra_layers.append(nn.Sequential(*layer))


# # VOC
# config = {
#     'backbone_layers': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
#     'extra_layers': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     'selected_layers': [4, 6, 6, 6, 4, 4],
#     'num_classes': 20,
#     # Prior box configs
#     'min_dim': 300,
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_sizes': [30, 60, 111, 162, 213, 264],
#     'max_sizes': [60, 111, 162, 213, 264, 315],
#     'steps': [8, 16, 32, 64, 100, 300],
#     'lr_steps': (80000, 100000, 120000),
#     'max_iter': 120000,
#     'clip': True,
#     'name': 'VOC'
# }



# class L2Norm(nn.Module):
#     def __init__(self, n_channels, scale):
#         super().__init__()
#         self.n_channels = n_channels
#         self.gamma = scale or None
#         self.eps = 1e-10
#         self.weight = nn.Parameter(torch.Tensor(self.n_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.constant_(self.weight, val=self.gamma)

#     def forward(self, x):
#         norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
#         #x /= norm
#         x = torch.div(x, norm)
#         out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x

#         return out


# class PriorBox:
#     """
#     Compute priorbox coordinates in center-offset form for each source feature map.
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.image_size = config['min_dim']
#         # number of priors for feature map location (either 4 or 6)
#         self.num_priors = len(config['aspect_ratios'])
#         self.variance = config['variance'] or [0.1]
#         self.feature_maps = config['feature_maps']
#         self.min_sizes = config['min_sizes']
#         self.max_sizes = config['max_sizes']
#         self.steps = config['steps']
#         self.aspect_ratios = config['aspect_ratios']
#         self.clip = config['clip']
#         self.version = config['name']

#         for v in self.variance:
#             if v <= 0:
#                 raise ValueError('Variances must be greater than 0')

#     def forward(self):
#         mean = []
#         for k, f in enumerate(self.feature_maps):
#             for i, j in itertools.product(range(f), repeat=2):
#                 f_k = self.image_size / self.steps[k]
#                 # unit center x,y
#                 cx = (j + 0.5) / f_k
#                 cy = (i + 0.5) / f_k

#                 # aspect_ratio: 1
#                 # rel size: min_size
#                 s_k = self.min_sizes[k] / self.image_size
#                 mean += [cx, cy, s_k, s_k]

#                 # aspect_ratio: 1
#                 # rel size: sqrt(s_k * s_(k+1))
#                 s_k_prime = math.sqrt(s_k * (self.max_sizes[k]/self.image_size))
#                 mean += [cx, cy, s_k_prime, s_k_prime]

#                 # rest of aspect ratios
#                 for ar in self.aspect_ratios[k]:
#                     mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
#                     mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
#         # back to torch land
#         output = torch.FloatTensor(mean).view(-1, 4)
#         if self.clip:
#             output.clamp_(max=1, min=0)
            
#         return output



        

# class SsdModel(nn.Module):
#     """Single Shot Multibox Architecture
#     The network is composed of a base VGG network followed by the
#     added multibox conv layers.  Each multibox layer branches into
#         1) conv2d for class conf scores
#         2) conv2d for localization predictions
#         3) associated priorbox layer to produce default bounding
#            boxes specific to the layer's feature map size.
#     See: https://arxiv.org/pdf/1512.02325.pdf for more details.
#     Args:
#         phase: (string) Can be "test" or "train"
#         size: input image size
#         base: VGG16 layers for input, size of either 300 or 500
#         extras: extra layers that feed to multibox loc and conf layers
#         head: "multibox head" consists of loc and conf conv layers
#     """
#     def __init__(self, phase, size, base, extras, head, num_classes):
#         super().__init__()
#         self.prediction_head = SsdPredictionHead()
#         self.phase = phase
#         self.num_classes = num_classes

#         self.cfg = (coco, voc)[num_classes == 21]


#         self.priorbox = PriorBox(self.cfg)

#         with torch.no_grad():
#             self.priors = self.priorbox.forward()
#             # self.priors = Variable(self.priorbox.forward(), volatile=True)
#         self.size = size

#         # SSD network
#         self.vgg = nn.ModuleList(base)
#         # Layer learns to scale the l2 normalized features from conv4_3
#         self.L2Norm = L2Norm(512, 20)
#         self.extras = nn.ModuleList(extras)

#         self.loc = nn.ModuleList(head[0])
#         self.conf = nn.ModuleList(head[1])

#         if phase == 'test':
#             self.softmax = nn.Softmax(dim=-1)
#             self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

#     def forward(self, x):
#         """Applies network layers and ops on input image(s) x.
#         Args:
#             x: input image or batch of images. Shape: [batch,3,300,300].
#         Return:
#             Depending on phase:
#             test:
#                 Variable(tensor) of output class label predictions,
#                 confidence score, and corresponding location predictions for
#                 each object detected. Shape: [batch,topk,7]
#             train:
#                 list of concat outputs from:
#                     1: confidence layers, Shape: [batch*num_priors,num_classes]
#                     2: localization layers, Shape: [batch,num_priors*4]
#                     3: priorbox layers, Shape: [2,num_priors*4]
#         """
#         sources = list()
#         loc = list()
#         conf = list()

#         # apply vgg up to conv4_3 relu
#         for k in range(23):
#             x = self.vgg[k](x)

#         s = self.L2Norm(x)
#         sources.append(s)

#         # apply vgg up to fc7
#         for k in range(23, len(self.vgg)):
#             x = self.vgg[k](x)
#         sources.append(x)

#         # apply extra layers and cache source layer outputs
#         for k, v in enumerate(self.extras):
#             x = F.relu(v(x), inplace=True)
#             if k % 2 == 1:
#                 sources.append(x)

#         # apply multibox head to source layers
#         for (x, l, c) in zip(sources, self.loc, self.conf):
#             loc.append(l(x).permute(0, 2, 3, 1).contiguous())
#             conf.append(c(x).permute(0, 2, 3, 1).contiguous())

#         loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
#         conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
#         if self.phase == "test":
#             output = self.detect(
#                 loc.view(loc.size(0), -1, 4),                   # loc preds
#                 self.softmax(conf.view(conf.size(0), -1,
#                              self.num_classes)),                # conf preds
#                 self.priors.type(type(x.data))                  # default boxes
#             )
#         else:
#             output = (
#                 loc.view(loc.size(0), -1, 4),
#                 conf.view(conf.size(0), -1, self.num_classes),
#                 self.priors
#             )
#         return output









