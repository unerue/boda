from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F


import torch
import torch.nn as nn
import os
import math
from collections import deque
from pathlib import Path
from layers.interpolate import InterpolateModule



def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """
    def make_layer(layer_cfg):
        nonlocal in_channels
        
        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels



class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

prior_cache = defaultdict(lambda: None)

class FPN(nn.Module):
    """Feature Pyramid Networks (FPN) for YOLACT
    https://arxiv.org/pdf/1612.03144.pdf
    """
    def __init__(self, in_channels: List = [512, 1024, 2048]):
        super().__init__()
        self.fpn_num_features = 256  # REPLACE CONFIG!
        self.fpn_pad = 1  # REPLACE CONFIG!
        self.fpn_num_downsample = 2

        # 1 x 1 conv to backbone feature map
        # ModuleList((0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        #            (1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        #            (2): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)))
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(x, self.fpn_num_features, kernel_size=1) for x in reversed(in_channels)])

        self.pred_layers = nn.ModuleList([
            nn.Conv2d(
                self.fpn_num_features, 
                self.fpn_num_features, 
                kernel_size=3, 
                padding=self.fpn_pad) for _ in in_channels])

        fpn_use_conv_downsample = True  # REPLACE CONFIG!
        if fpn_use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(
                    self.fpn_num_features, 
                    self.fpn_num_features,
                    kernel_size=3, 
                    stride=2,
                    padding=1) for _ in range(self.fpn_num_downsample)])

        self.interpolate_mode = 'bilinear'

    def forward(self, inputs: List[Tensor]):
        """
        backbone_outs = [[n, 512, 69, 69], [n, 1024, 35, 35], [n, 2048, 18, 18]]
        In class Yolact's train(), remove C2 from bakebone_outs. So FPN gets three feature outs.
        """
        outputs = []
        x = torch.zeros(1, device=inputs[0].device)
        for _ in range(len(inputs)):
            outputs.append(x)

        j = len(inputs)
        for lateral_layer in self.lateral_layers:
            j -= 1
            if j < len(inputs) - 1:
                _, _, h, w = inputs[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolate_mode, align_corners=False)
            
            x = x + lateral_layer(inputs[j])
            outputs[j] = x

        j = len(inputs)
        for pred_layer in self.pred_layers:
            j -= 1
            outputs[j] = F.relu(pred_layer(outputs[j]))

        for downsample_layer in self.downsample_layers:
            outputs.append(downsample_layer(outputs[-1]))

        return outputs


class YolactPredictionHead(nn.Module):
    def __init__(self, config, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()
        
        self.num_classes = 20 # cfg.num_classes
        self.mask_dim    = None # cfg.mask_dim # Defined by Yolact
        self.num_priors  = sum(len(x)*len(scales) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index
        self.num_heads   = cfg.num_heads # Defined by Yolact

        mask_proto_split_prototypes_by_head = False
        mask_type = 1
        mask_type_linecomb = 1
        # if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
        if mask_proto_split_prototypes_by_head and mask_type == mask_type_linecomb:
            self.mask_dim = self.mask_dim // self.num_heads

        mask_proto_prototypes_as_features = False
        # if cfg.mask_proto_prototypes_as_features:
        if mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        
        extra_head_net = None
        if parent is None:
            # if cfg.extra_head_net is None:
            if extra_head_net is None:
                out_channels = in_channels
            else:
                # self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)
                self.upfeature, out_channels = make_net(in_channels, extra_head_net)

            use_prediction_module = False
            # if cfg.use_prediction_module:
            if use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)

            head_layer_params = {'kernel_size': 3, 'padding': 1},
            # self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **cfg.head_layer_params)
            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **head_layer_params)
            
            use_mask_scoring = False
            # if cfg.use_mask_scoring:
            if use_mask_scoring:
                # self.score_layer = nn.Conv2d(out_channels, self.num_priors, **cfg.head_layer_params)
                self.score_layer = nn.Conv2d(out_channels, self.num_priors, **head_layer_params)

            use_instance_coeff = False
            num_instance_coeffs = 64
            # if cfg.use_instance_coeff:
            if use_instance_coeff:
                # self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * num_instance_coeffs, **head_layer_params)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
            
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        src = self if self.parent[0] is None else self.parent[0]
        
        conv_h = x.size(2)
        conv_w = x.size(3)

        extra_head_net = [(256, 3, {'padding': 1})]
        # if cfg.extra_head_net is not None:
        if extra_head_net is not None:
            x = src.upfeature(x)
        
        use_prediction_module = False
        # if cfg.use_prediction_module:
        if use_prediction_module:
            # The two branches of PM design (c)
            a = src.block(x)
            
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        
        eval_mask_branch = True
        # if cfg.eval_mask_branch:
        if eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)
        
        use_mask_scoring = False
        # if cfg.use_mask_scoring:
        if use_mask_scoring:
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        use_instance_coeff = False
        num_instance_coeffs = 64
        # if cfg.use_instance_coeff:
        if use_instance_coeff:
            # inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, num_instance_coeffs)

        # See box_utils.decode for an explanation of this
        use_yolo_regressors = False
        # if cfg.use_yolo_regressors:
        if use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        eval_mask_branch = True
        mask_type_direct = 0
        # if cfg.eval_mask_branch:
        if eval_mask_branch:
            # if cfg.mask_type == mask_type.direct:
            if mask_type == mask_type_direct:
                mask = torch.sigmoid(mask)
            # elif cfg.mask_type == mask_type.lincomb:
            elif mask_type == mask_type_lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)

                mask_proto_coeff_gate = False
                # if cfg.mask_proto_coeff_gate:
                if mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)

        mask_proto_split_prototypes_by_head = False
        # if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
        if mask_proto_split_prototypes_by_head and mask_type == mask_type_lincomb:
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim), mode='constant', value=0)
        
        priors = self.make_priors(conv_h, conv_w, x.device)

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }

        use_mask_scoring = False
        # if cfg.use_mask_scoring:
        if use_mask_scoring:
            preds['score'] = score

        use_instance_coeff = False
        # if cfg.use_instance_coeff:
        if use_instance_coeff:
            preds['inst'] = inst
        
        return preds

    def make_priors(self, conv_h, conv_w, device):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        global prior_cache
        size = (conv_h, conv_w)

        # with timer.env('makepriors'):
            # if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
                # prior_data = []
        _tmp_img_w = ?
        _tmp_img_h = ?
        # if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
        if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
            prior_data = []

            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h
                
                for ars in self.aspect_ratios:
                    for scale in self.scales:
                        for ar in ars:
                            preapply_sqrt = 
                            # if not cfg.backbone.preapply_sqrt:
                            if not backbone.preapply_sqrt:
                                ar = sqrt(ar)

                            use_pixel_scales = 
                            max_size = 
                            # if cfg.backbone.use_pixel_scales:
                            if backbone.use_pixel_scales:
                                # w = scale * ar / cfg.max_size
                                w = scale * ar / max_size
                                h = scale / ar / max_size
                            else:
                                w = scale * ar / conv_w
                                h = scale / ar / conv_h
                            
                            # This is for backward compatability with a bug where I made everything square by accident
                            backbone_use_square_anchors
                            # if cfg.backbone.use_square_anchors:
                            if backbone.use_square_anchors:
                                h = w

                            prior_data += [x, y, w, h]

            self.priors = torch.Tensor(prior_data, device=device).view(-1, 4).detach()
            self.priors.requires_grad = False
            # self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)
            self.last_img_size = (_tmp_img_w, _tmp_img_h)
            self.last_conv_size = (conv_w, conv_h)
            prior_cache[size] = None

        elif self.priors.device != device:
            # This whole weird situation is so that DataParalell doesn't copy the priors each iteration
            if prior_cache[size] is None:
                prior_cache[size] = {}
            
            if device not in prior_cache[size]:
                prior_cache[size][device] = self.priors.to(device)

            self.priors = prior_cache[size][device]
        
        return self.priors

class FastMaskIoUNet(ScriptModuleWrapper):
    def __init__(self):
        super().__init__()
        input_channels = 1
        last_layer = [(cfg.num_classes-1, 1, {})]
        
        maskiou_net = 
        # self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)
        self.maskiou_net, _ = make_net(input_channels, maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

        return maskiou_p


class YolactModel:
    def __init__(self, backbone):
        self.backbone = backbone
        self.fpn = FPN([512, 1024, 2048])
        backbone_outs=[]
        for i in [1,2,3]:
            backbone_outs.append(self.backbone[i])

        self.fpn_outs = self.fpn(backbone_outs)










class Yolact(nn.Module):
    """
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
    You can set the arguments by changing them in the backbone config object in config.py.
    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """
    def __init__(self):
        super().__init__()
        # cfg.backbone
        self.backbone = construct_backbone(cfg.backbone)

        cfg_freeze_bn:
        # if cfg.freeze_bn:
        if cfg_freeze_bn:
            self.freeze_bn()


        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        cfg_mask_type = False
        mask_type_direct = 0
        cfg_mask_dim = None
        cfg_mask_size = None
        # if cfg.mask_type == mask_type.direct:
        if mask_type == mask_type_direct:
            # cfg.mask_dim = cfg.mask_size**2
            cfg_mask_dim = cfg_mask_size**2

        # elif cfg.mask_type == mask_type.lincomb:
        elif cfg_mask_type == mask_type_lincomb:
            cfg_mask_proto_use_grid = True
            # if cfg.mask_proto_use_grid:
            if cfg_mask_proto_use_grid:
                cfg_mask_proto_grid_file = None
                # self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.grid = torch.Tensor(np.load(cfg_mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            cfg_mask_proto_src = None
            # self.proto_src = cfg.mask_proto_src
            self.proto_src = cfg.mask_proto_src

            cfg_fpn = None
            cfg_fpn_num_features = None
            if self.proto_src is None: 
                in_channels = 3
            # elif cfg.fpn is not None: 
            elif cfg_fpn is not None: 
                # in_channels = cfg.fpn.num_features
                in_channels = cfg_fpn_num_features
            else: 
                in_channels = self.backbone.channels[self.proto_src]

            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            cfg_mask_proto_net = None
            # self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg_mask_proto_net, include_last_relu=False)

            # if cfg.mask_proto_bias:
            #     cfg.mask_dim += 1
            cfg_mask_proto_bias = False
            cfg_mask_dim = 0
            if cfg.mask_proto_bias:
                cfg_mask_dim += 1


        cfg_backbone_selected_layers = None
        # self.selected_layers = cfg.backbone.selected_layers
        self.selected_layers = cfg_backbone_selected_layers
        src_channels = self.backbone.channels

        cfg_use_maskiou = False
        # if cfg.use_maskiou:
        if cfg.use_maskiou:
            self.maskiou_net = FastMaskIoUNet()


        # if cfg.fpn is not None:
        if cfg_fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])
            cfg_fpn_num_downsample = None
            # self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))

            cfg_fpn.num_features = 5
            # src_channels = [cfg.fpn.num_features] * len(self.selected_layers)
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)


        self.prediction_layers = nn.ModuleList()
        # cfg.num_heads = len(self.selected_layers)
        cfg_num_heads = len(self.selected_layers)

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            # if cfg.share_prediction_module and idx > 0:
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            # pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
            #                         aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
            #                         scales        = cfg.backbone.pred_scales[idx],
            #                         parent        = parent,
            #                         index         = idx)

            pred = YolactPredictionHead(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios=cfg_backbone_pred_aspect_ratios[idx],
                                    scales=cfg_backbone_pred_scales[idx],
                                    parent=parent,
                                    index=idx)
            self.prediction_layers.append(pred)

        # Extra parameters for the extra losses
        # if cfg.use_class_existence_loss:
        cfg_num_classes = None
        cfg_use_class_existence_loss = False
        if cfg_use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            # self.class_existence_fc = nn.Linear(src_channels[-1], cfg_num_classes - 1)
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        
        if cfg_use_semantic_segmentation_loss:
        # if cfg.use_semantic_segmentation_loss:
        if cfg_use_semantic_segmentation_loss:
            # self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes-1, kernel_size=1)
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg_num_classes-1, kernel_size=1)

        # For use in evaluation
        cfg_num_classes = 20
        cfg_nms_top_k = None
        cfg_nms_conf_thresh
        cfg_nms_thresh
        # self.detect = Detect(
        #     cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
        #     conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)
        self.detect = Detect(
            cfg_num_classes, bkg_label=0, top_k=cfg_nms_top_k,
            conf_thresh=cfg_nms_conf_thresh, nms_thresh=cfg_nms_thresh)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            cfg_fpn = 
            cfg_fpn_num_downsample = 
            if key.startswith('fpn.downsample_layers.'):
                # if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg_fpn_num_downsample:
                    del state_dict[key]
        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')
        
        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))
            
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            cfg_use_focal_loss = 
            cfg_use_sigmoid_focal_loss = 
            cfg_focal_loss_init_pi = 
            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    # if cfg.use_focal_loss and 'conf_layer' in name:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        # if not cfg.use_sigmoid_focal_loss:
                        if not cfg_use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0

                            # module.bias.data[0]  = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[0]  = np.log((1 - cfg_focal_loss_init_pi) / cfg_focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            # module.bias.data[0]  = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            # module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[0]  = -np.log(cfg_focal_loss_init_pi / (1 - cfg_focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg_focal_loss_init_pi) / cfg_focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()
    
    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
    
    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        _, _, img_h, img_w = x.size()
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w
        
        with timer.env('backbone'):
            outs = self.backbone(x)

        if cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs = self.fpn(outs)

        proto_out = None
        # if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]
                
                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                # proto_out = cfg.mask_proto_prototype_activation(proto_out)
                proto_out = cfg.mask_proto_prototype_activation(proto_out)

                # if cfg.mask_proto_prototypes_as_features:
                if cfg.mask_proto_prototypes_as_features:
                    # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                    proto_downsampled = proto_out.clone()
                    # if cfg.mask_proto_prototypes_as_features_no_grad:
                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                
                # Move the features last so the multiplication is easy
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)


        # with timer.env('pred_heads'):
        #     pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }

        #     if cfg.use_mask_scoring:
        #         pred_outs['score'] = []
        
        pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }
        # if cfg.use_mask_scoring:
        if cfg.use_mask_scoring:
            pred_outs['score'] = []

        # if cfg.use_instance_coeff:
        if cfg.use_instance_coeff:
            pred_outs['inst'] = []
        
        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]

            # if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                # Scale the prototypes down to the current prediction layer's size and add it as inputs
                proto_downsampled = F.interpolate(
                    proto_downsampled,
                    size=outs[idx].size()[2:], 
                    mode='bilinear', 
                    align_corners=False)

                pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

            # A hack for the way dataparallel works
            # if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
            if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                pred_layer.parent = [self.prediction_layers[0]]

            p = pred_layer(pred_x)
            
            for k, v in p.items():
                pred_outs[k].append(v)

        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        if self.training:
            # For the extra loss functions
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            return pred_outs
        else:
            if cfg.use_mask_scoring:
                pred_outs['score'] = torch.sigmoid(pred_outs['score'])

            if cfg.use_focal_loss:
                if cfg.use_sigmoid_focal_loss:
                    # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                    pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
                    if cfg.use_mask_scoring:
                        pred_outs['conf'] *= pred_outs['score']
                elif cfg.use_objectness_score:
                    # See focal_loss_sigmoid in multibox_loss.py for details
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                    pred_outs['conf'][:, :, 0 ] = 1 - objectness
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            else:

                if cfg.use_objectness_score:
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    
                    pred_outs['conf'][:, :, 1:] = (objectness > 0.10)[..., None] \
                        * F.softmax(pred_outs['conf'][:, :, 1:], dim=-1)
                    
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

            return self.detect(pred_outs, self)



if __name__ == '__main__':
    from torchsummary import summary
    from backbone.backbone_yolact import YolactBackbone

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = YolactBackbone().to(device)
    print(summary(backbone, input_data=(3, 550, 550), verbose=0))
    
    input_data = torch.randn(1, 3, 550, 550)
    backbone = construct_backbone()(input_data)

    backbone_selected_layers = [1, 2, 3]
    backbone_outs=[]
    for i in [1,2,3] :
        backbone_outs.append(backbone[i])

    fpn = FPN([512, 1024, 2048])
    fpn_outs = fpn(backbone_outs)

    print(f'P3 shape: {fpn_outs[0].size()}')
    print(f'P4 shape: {fpn_outs[1].size()}')
    print(f'P5 shape: {fpn_outs[2].size()}')
    print(f'P6 shape: {fpn_outs[3].size()}')
    print(f'P7 shape: {fpn_outs[4].size()}')
    print(f'Number of FPN output feature: {len(fpn_outs)}')