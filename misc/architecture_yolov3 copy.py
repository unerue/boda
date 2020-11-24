import math
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from backbones import backbone_fn
# from collections import OrderedDict
# from misc.nms.nms_wrapper import nms
# import pdb
from backbone import darknet53
from configuration import yolov3_base_darknet_pascal



# def conv1x1(in_channels, out_channels):
#     return 


class Conv2d1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels[0], 
            kernel_size=1,
            stride=1, 
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(
            out_channels[0], 
            out_channels[1], 
            kernel_size=3,
            stride=1, 
            padding=1, 
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Yolov3Model:
    """YOLOv3 architecture

    Yolov3PredictionHead ->
    Yolov3Loss(inputs, labels)
    
    """
    def __init__(self):
        raise NotImplementedError


class Yolov3PredictionHead(nn.Module):
    """ Yolov3 Prediction Head

    백본에서 아웃풋을 받아서 로스 펑션에 넣을 수 있게 변환해주는 헤드
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        print(self.config.print())
        
        self.backbone = darknet53()  # self.config.backbone
        
        out_channels = self.backbone.channels[-self.config.selected_layers:]  ########## CONFIG!!!!!!
        print(out_channels)
        print(out_channels[1:])
        # C = 80  ########## CONFIG!!!!!!
        # B = 3   ########## CONFIG!!!!!!
        # N × N × [3 ∗ (4 + 1 + 80)]
        num_features = self.config.num_boxes * (4+1+self.config.dataset.num_classes)
        print(num_features)


        self.detect_layers = nn.ModuleList()
        for in_channels in out_channels:
            self.detect_layers.append(Conv2d1x1(in_channels, num_features))

        self.upsample_layers = nn.ModuleList()
        for in_channels in out_channels[1:]:
            self.upsample_layers.append(
                nn.Sequential(
                    Conv2d1x1(in_channels, int(in_channels*0.25)),
                    Upsample(2)))

        self.shortcut_layers = nn.ModuleList()
        for in_channels, downsample in zip(out_channels[1:], self.config.downsample):
            _in_channels = [downsample[1]] * (self.config.selected_layers-1)
            _in_channels.insert(0, in_channels - int(in_channels*0.25))
            print(_in_channels)
            self.shortcut_layers.append(
                nn.Sequential(*[BasicBlock(_in, downsample) for _in in _in_channels]))
            

            # print(in_channels, downsample)

        self.out1_conv1x1 = Conv2d1x1(256, num_features)
        self.out2_conv1x1 = Conv2d1x1(512, num_features)
        self.out3_conv1x1 = Conv2d1x1(1024, num_features)

        self.out3_upsample = nn.Sequential(
            Conv2d1x1(out_channels[-1], int(out_channels[-1]*0.25)),
            Upsample(2))

        self.out2_upsample = nn.Sequential(
            Conv2d1x1(out_channels[-2], int(out_channels[-2]*0.25)),
            Upsample(2))

        self.out2_layer = nn.Sequential(*[BasicBlock(_in, [256, 512]) for _in in [768, 512, 512]])
        self.out1_layer = nn.Sequential(*[BasicBlock(_in, [128, 256]) for _in in [384, 256, 256]])
        

        self.out1_conv1x1 = Conv2d1x1(256, 255)
        self.out2_conv1x1 = Conv2d1x1(512, 255)
        self.out3_conv1x1 = Conv2d1x1(1024, 255)

        # self.upsample_layer = nn.ModuleList([
        #         nn.Conv2d(
        #             self.fpn_num_features, 
        #             self.fpn_num_features,
        #             kernel_size=1, 
        #             stride=1,
        #             padding=0) for _ in range(3)])
    # [512, 1, 1, 1], [1024, 3, 1, 1], [512, 1, 1, 1], [1024, 3, 1, 1], [512, 1, 1, 1], [1024, 3, 1, 1]
    # [255, 1, 1, 1]
    
    
    # [256, 512]
    # [128, 256]

    def forward(self, x):
        features = self.backbone(x)[-self.config.selected_layers:]
        j = len(features)
        outs = []
        for feature in reversed(features):
            if j < len(features):
                concat_out = torch.cat([feature, residual], 1)
                # print(concat_out.size(), j, '** CONCATENATE!!')
                features = self.shortcut_layers[j-1](concat_out)
                # print(out.size(), j, '** CONCAT + DETECTION LAYER!!!')
            j -= 1
            # if j != 0:
            out = self.detect_layers[j](feature)
            # print(out.size(), j, '******* DETECTION !!!!')
            if j != 0:
                residual = self.upsample_layers[j-1](feature)
                # print(residual.size(), j, '** UPSAMPLE!!')
            outs.append(out)
            

        for out in outs:
            print(out.size())
            
            

        print('='*50)
        # print(len(outs))
        
        # layer = nn.Sequential(*[BasicBlock(256, [512, 1024]) for _ in range(3)])
        # print(layer)
        # outs.reverse()
        # for i, out in enumerate(outs):
        #     if i+1 < len(outs):
        #         print('original', out.size())
        #         print('next', outs[i+1].size())
        #         down_out = self.conv1x1(out)
                
                
        #         print('downsample', down_out.size())
        #         up_out = Upsample(2)(out)
        #         print('upsample', up_out.size())
        out1, out2, out3 = self.backbone(x)[2:]
        print(out1.size())
        print(out2.size())
        print(out3.size())
        
        out3_upsample = self.out3_upsample(out3)
        out3 = self.out3_conv1x1(out3)
        print(out3_upsample.size())
        print('Predictions:', out3.size())
        print()
        # Out2
        out2_concat = torch.cat([out2, out3_upsample], 1)
        print(out2_concat.size())

        out2_concat = self.out2_layer(out2_concat)
        print(out2_concat.size())

        out2_upsample = self.out2_upsample(out2_concat)
        print(out2_upsample.size())

        out2 = self.out2_conv1x1(out2_concat)
        print('Predictions:', out2.size())
        print()

        out1_concat = torch.cat([out1, out2_upsample], 1)
        print(out1_concat.size())

        out1 = self.out1_layer(out1_concat)
        print(out1.size())

        out1 = self.out1_conv1x1(out1)
        print('Predictions:', out1.size())



        



        return out3
    
    # def _make_layer(self):
import numpy as np


        


# loss 함수 제일 첫번째 
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):

    num_anchors = len(anchors)

    anchors_tensor = torch.from_numpy(anchors).view(1,1,1,num_anchors, 2).type_as(feats)

    grid_shape = (feats.shape[2:4])

    grid_y = torch.arange(0, grid_shape[0]).view(-1, 1, 1, 1).expand(grid_shape[0], grid_shape[0], 1, 1)
    grid_x = torch.arange(0, grid_shape[1]).view(1, -1, 1, 1).expand(grid_shape[1], grid_shape[1], 1, 1)

    grid = torch.cat([grid_x, grid_y], dim=3).unsqueeze(0).type_as(feats)

    feats = feats.view(-1, num_anchors, num_classes+5, grid_shape[0], \
                grid_shape[1]).permute(0, 3, 4, 1, 2).contiguous()

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (torch.sigmoid(feats[...,:2]) + grid) / torch.tensor(grid_shape).view(1,1,1,1,2).type_as(feats)
    box_wh = torch.exp(feats[..., 2:4]) * anchors_tensor / input_shape.view(1,1,1,1,2)

    box_confidence = torch.sigmoid(feats[..., 4:5])
    box_class_probs = torch.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_probs



def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = torch.FloatTensor(nB, nA, nG, nG).fill_(0)

    iou_scores = torch.FloatTensor(nB, nA, nG, nG).fill_(0)

    tx = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    th = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


class Yolov3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        anchors = [
            (10, 13), (16, 30), (33, 23), 
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)
        ]
        self.anchors = np.array(anchors)
        self.num_layers = len(self.anchors) // 3

        self.num_anchors = len(anchors)

        self.anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if self.num_layers == 3 else [[3,4,5], [1,2,3]]
        self.num_classes = 80
        self.ignore_thresh = 0.5

        self.mse_loss = nn.MSELoss(reduction='sum')
        
        self.grid_size = 0

        self.img_dim = 416

    def _grid_offset(self, grid_size, cuda=True):
        """
        grid_size = 박스 높이 52 x 52, 26 x26 등
        """
        self.grid_size = grid_size
        g = self.grid_size

        self.stride = self.img_dim / self.grid_size

        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]) #.type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g])

        self.scaled_anchors = torch.FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        return self

    def forward(self, x, target=None):
        num_samples = x.size(0) # Batch
        grid_size = x.size(2) # or size(3)

        preds = (
            x.view(num_samples, self.num_anchors, self.num_classes+5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(preds[..., 0])  # Center x
        y = torch.sigmoid(preds[..., 1])  # Center y
        w = preds[..., 2]  # Width
        h = preds[..., 3]  # Height
        pred_conf = torch.sigmoid(preds[..., 4])  # Conf
        pred_cls = torch.sigmoid(preds[..., 5:])  # Cls pred.

        if grid_size != self.grid_size:
            self._grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(preds[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        
        ######################
        # Yolov3 header 
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
        )

        ############################################# LOSS FUNCTION ################

        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask]) # nn.MSELoss
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])

        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            'loss': to_cpu(total_loss).item(),
            'x': to_cpu(loss_x).item(),
            'y': to_cpu(loss_y).item(),
            'w': to_cpu(loss_w).item(),
            'h': to_cpu(loss_h).item(),
            'conf': to_cpu(loss_conf).item(),
            'cls': to_cpu(loss_cls).item(),
            'cls_acc': to_cpu(cls_acc).item(),
            'recall50': to_cpu(recall50).item(),
            'recall75': to_cpu(recall75).item(),
            'precision': to_cpu(precision).item(),
            'conf_obj': to_cpu(conf_obj).item(),
            'conf_noobj': to_cpu(conf_noobj).item(),
            'grid_size': grid_size,
        }

        return output, total_loss




        input_shape = torch.Tensor([
            target.size(2) * 32, 
            target.size(3) * 32]).type_as(target[0]) 
        
        grid_shapes = [
            torch.Tensor([output.size(2), output.size(3)]).type_as(target[0]) 
            for output in target] 
        
        m = target[0].size(0)


        loss_xy = 0
        loss_wh = 0
        loss_conf = 0
        loss_clss = 0
        nRecall = 0
        nRecall75 = 0
        nProposal = 0

        for l in range(self.num_layers):

            object_mask = y_true[l][..., 4:5]
            true_class_probs = y_true[l][..., 5:]
            grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], 
                                self.anchors[self.anchor_mask[l]], self.num_classes, input_shape, calc_loss=True)            

            pred_box = torch.cat([pred_xy, pred_wh], dim=4)
            # Darknet raw box to calculate loss.
            raw_true_xy = y_true[l][..., :2]*grid_shapes[l].view(1,1,1,1,2) - grid
            raw_true_wh = torch.log(y_true[l][..., 2:4] / torch.Tensor(self.anchors[self.anchor_mask[l]]).type_as(pred_box).view(1,1,1,self.num_layers,2) * input_shape.view(1,1,1,1,2))
            raw_true_wh.masked_fill_(object_mask.expand_as(raw_true_wh)==0, 0)
            box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

            # Find ignore mask, iterate over each of batch.
            # ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            best_ious = []
            for b in range(m):
                true_box = y_true[l][b,...,0:4][object_mask[b,...,0]==1]
                iou = box_iou(pred_box[b], true_box)
                best_iou, _ = torch.max(iou, dim=3)
                best_ious.append(best_iou)
            
            best_ious = torch.stack(best_ious, dim=0).unsqueeze(4)
            ignore_mask = (best_ious < self.ignore_thresh).float()

            # binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = self.mse_loss(object_mask*box_loss_scale*torch.sigmoid(raw_pred[...,0:2]), object_mask * raw_true_xy)/ m
            wh_loss = torch.sum(object_mask * box_loss_scale * 0.5 * (raw_true_wh-raw_pred[...,2:4])**2)/m

            confidence_loss = (self.mse_loss(torch.sigmoid(raw_pred[...,4:5])[object_mask == 1], object_mask[object_mask==1]) + \
                            self.mse_loss(torch.sigmoid(raw_pred[...,4:5])[((1-object_mask)*ignore_mask) == 1], object_mask[((1-object_mask)*ignore_mask) == 1]))/m

            class_loss = self.mse_loss(torch.sigmoid(raw_pred[...,5:])* object_mask, true_class_probs * object_mask)/m

            loss_xy += xy_loss
            loss_wh += wh_loss
            loss_conf += confidence_loss
            loss_clss += class_loss
            # loss += xy_loss + wh_loss + confidence_loss + class_loss
        
            nRecall += torch.sum(best_ious > 0.5)
            nRecall75 += torch.sum(best_ious > 0.75)
            nProposal += torch.sum(torch.sigmoid(raw_pred[...,4:5]) > 0.25)





    def _make_cbl(self, in_channels, out_channels, kernel_size):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        padding = (kernel_size - 1) // 2 if ks else 0
        return nn.Sequential([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(_out),
            nn.LeakyReLU(0.1),
        ])

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module('conv_out', nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m



    


class YOLOv3(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.config = config

        self.backbone = backbone_fn(opt)
        _out_filters = self.backbone.layers_out_filters

        final_out_filter0 = 3 * (5 + opt.classes)

        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = 3 * (5 + opt.classes)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        # self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = 3 * (5 + opt.classes)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        # self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

        self.anchors = np.array(opt.anchors)
        self.num_layers = len(self.anchors) // 3
        self.num_classes = opt.classes


        # initlize the loss function here.
        self.loss = yolo_loss(opt)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def _branch(self, _embedding, _in):
        for i, e in enumerate(_embedding):
            _in = e(_in)
            if i == 4:
                out_branch = _in
        return _in, out_branch

    def forward(self, img, label1, label2, label3):

        if self.opt.backbone_lr == 0:
            with torch.no_grad():
                x2, x1, x0 = self.backbone(img)
        else:
            x2, x1, x0 = self.backbone(img)

        out0, out0_branch = self._branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = F.interpolate(x1_in, scale_factor=2, mode='nearest')
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = self._branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        # x2_in = self.embedding2_upsample(x2_in)
        x2_in = F.interpolate(x2_in, scale_factor=2, mode='nearest')
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = self._branch(self.embedding2, x2_in)

        loss = self.loss((out0, out1, out2), (label1, label2, label3))

        return loss


    def detect(self, img, ori_shape):

        with torch.no_grad():
            x2, x1, x0 = self.backbone(img)
            # forward the decoder block
            out0, out0_branch = self._branch(self.embedding0, x0)
            #  yolo branch 1
            x1_in = self.embedding1_cbl(out0_branch)
            x1_in = F.interpolate(x1_in, scale_factor=2, mode='nearest')
            x1_in = torch.cat([x1_in, x1], 1)
            out1, out1_branch = self._branch(self.embedding1, x1_in)
            #  yolo branch 2
            x2_in = self.embedding2_cbl(out1_branch)
            x2_in = F.interpolate(x2_in, scale_factor=2, mode='nearest')
            x2_in = torch.cat([x2_in, x2], 1)
            out2, out2_branch = self._branch(self.embedding2, x2_in)

        image_shape = torch.Tensor([img.size(2), img.size(3)]).type_as(img)
        boxes_, scores_, classes_  = yolo_eval((out0, out1, out2), self.anchors, self.num_classes, image_shape, ori_shape)

        return boxes_, scores_, classes_

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)

    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = boxes.view([-1, 4])

    box_scores = box_confidence * box_class_probs
    box_scores = box_scores.view(-1, num_classes)
    return boxes.view(feats.size(0), -1,4), box_scores.view(feats.size(0), -1,num_classes)

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''

    box_yx = torch.stack((box_xy[...,1], box_xy[...,0]), dim=4)
    box_hw = torch.stack((box_wh[...,1], box_wh[...,0]), dim=4)

    new_shape = torch.round(image_shape * torch.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = torch.stack([
        box_mins[..., 0],  # y_min
        box_mins[..., 1],  # x_min
        box_maxes[..., 0],  # y_max
        box_maxes[..., 1]  # x_max
    ], dim=4)

    # Scale boxes back to original image shape.
    boxes *= torch.cat([image_shape, image_shape]).view(1,1,1,1,4)
    return boxes

def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              ori_shape,
              max_boxes=20,
              score_threshold=.5,
              iou_threshold=.5,
              nms_threshold=.3):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    max_per_image = 100
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = torch.Tensor([yolo_outputs[0].shape[2] * 32, yolo_outputs[0].shape[3] * 32]).type_as(yolo_outputs[0]) 
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)        
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = torch.cat(boxes, dim=1)
    box_scores = torch.cat(box_scores, dim=1)

    dets_ = []
    classes_ = []
    images_ = []
    for i in range(boxes.size(0)):
        mask = box_scores[i] >= score_threshold
        img_dets = []
        img_classes = []
        img_images = []

        for c in range(num_classes):
            class_boxes = boxes[i][mask[:,c]]
            if len(class_boxes) == 0:
                continue

            class_box_scores = box_scores[i][:,c][mask[:,c]]
            _, order = torch.sort(class_box_scores, 0, True)
            # do nms here.
            cls_dets =  torch.cat((class_boxes, class_box_scores.view(-1,1)), 1)
            cls_dets = cls_dets[order]

            keep = nms(cls_dets, nms_threshold)
            cls_dets = cls_dets[keep.view(-1).long()]

            img_dets.append(cls_dets)
            img_classes.append(torch.ones(cls_dets.size(0)) * c)
            img_images.append(torch.ones(cls_dets.size(0)) * i)

        # Limit to max_per_image detections *over all classes*
        if len(img_dets) > 0:
            img_dets = torch.cat(img_dets, dim=0)
            img_classes = torch.cat(img_classes, dim=0)
            img_images = torch.cat(img_images, dim=0)

            if max_per_image > 0:
                if img_dets.size(0) > max_per_image:
                    _, order = torch.sort(img_dets[:,4], 0, True)
                    keep = order[:max_per_image]
                    img_dets = img_dets[keep]
                    img_classes = img_classes[keep]
                    img_images = img_images[keep]

            # conver the bounding box back.
            w, h = image_shape[0].item(), image_shape[1].item()
            iw, ih = ori_shape[i][0].item(), ori_shape[i][1].item()

            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)            
            
            dx = (w-nw)//2
            dy = (h-nh)//2            

            img_dets[:, [0,2]] = (img_dets[:, [0,2]] - dx) / scale
            img_dets[:, [1,3]] = (img_dets[:, [1,3]] - dy) / scale
            img_classes[:] = img_classes[:] + 1 # since the evaluation need to start from 1

            dets_.append(img_dets)
            classes_.append(img_classes)
            images_.append(img_images)


    dets_ = torch.cat(dets_, dim=0)
    images_ = torch.cat(images_, dim=0)
    classes_ = torch.cat(classes_, dim=0)

    return dets_, images_, classes_

def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = b1.unsqueeze(3)

    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # if b2 is an empty tensor: then iou is empty
    if b2.shape[0] == 0:
        iou = torch.zeros(b1.shape[0:4]).type_as(b1)
    else:
        b2 = b2.view(1,1,1,b2.size(0), b2.size(1))
        # Expand dim to apply broadcasting.
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh/2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.clamp(intersect_maxes - intersect_mins, min=0)

        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):

    num_anchors = len(anchors)

    anchors_tensor = torch.from_numpy(anchors).view(1,1,1,num_anchors, 2).type_as(feats)

    grid_shape = (feats.shape[2:4])

    grid_y = torch.arange(0, grid_shape[0]).view(-1, 1, 1, 1).expand(grid_shape[0], grid_shape[0], 1, 1)
    grid_x = torch.arange(0, grid_shape[1]).view(1, -1, 1, 1).expand(grid_shape[1], grid_shape[1], 1, 1)

    grid = torch.cat([grid_x, grid_y], dim=3).unsqueeze(0).type_as(feats)

    feats = feats.view(-1, num_anchors, num_classes+5, grid_shape[0], \
                grid_shape[1]).permute(0, 3, 4, 1, 2).contiguous()

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (torch.sigmoid(feats[...,:2]) + grid) / torch.tensor(grid_shape).view(1,1,1,1,2).type_as(feats)
    box_wh = torch.exp(feats[..., 2:4]) * anchors_tensor / input_shape.view(1,1,1,1,2)

    box_confidence = torch.sigmoid(feats[..., 4:5])
    box_class_probs = torch.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

class yolo_loss(nn.Module):
    def __init__(self, opt):
        super(yolo_loss, self).__init__()

        self.opt = opt
        self.anchors = np.array(opt.anchors)
        self.num_layers = len(self.anchors) // 3
        self.anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if self.num_layers==3 else [[3,4,5], [1,2,3]]
        self.num_classes = opt.classes
        self.ignore_thresh = 0.5

        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, yolo_outputs, y_true):
        
        input_shape = torch.Tensor([yolo_outputs[0].shape[2] * 32, yolo_outputs[0].shape[3] * 32]).type_as(yolo_outputs[0]) 
        grid_shapes = [torch.Tensor([output.shape[2], output.shape[3]]).type_as(yolo_outputs[0]) for output in yolo_outputs] 
        m = yolo_outputs[0].size(0)

        loss_xy = 0
        loss_wh = 0
        loss_conf = 0
        loss_clss = 0
        nRecall = 0
        nRecall75 = 0
        nProposal = 0
        for l in range(self.num_layers):

            object_mask = y_true[l][..., 4:5]
            true_class_probs = y_true[l][..., 5:]
            grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], 
                                self.anchors[self.anchor_mask[l]], self.num_classes, input_shape, calc_loss=True)            

            pred_box = torch.cat([pred_xy, pred_wh], dim=4)
            # Darknet raw box to calculate loss.
            raw_true_xy = y_true[l][..., :2]*grid_shapes[l].view(1,1,1,1,2) - grid
            raw_true_wh = torch.log(y_true[l][..., 2:4] / torch.Tensor(self.anchors[self.anchor_mask[l]]).type_as(pred_box).view(1,1,1,self.num_layers,2) * input_shape.view(1,1,1,1,2))
            raw_true_wh.masked_fill_(object_mask.expand_as(raw_true_wh)==0, 0)
            box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

            # Find ignore mask, iterate over each of batch.
            # ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            best_ious = []
            for b in range(m):
                true_box = y_true[l][b,...,0:4][object_mask[b,...,0]==1]
                iou = box_iou(pred_box[b], true_box)
                best_iou, _ = torch.max(iou, dim=3)
                best_ious.append(best_iou)
            
            best_ious = torch.stack(best_ious, dim=0).unsqueeze(4)
            ignore_mask = (best_ious < self.ignore_thresh).float()

            # binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = self.mse_loss(object_mask*box_loss_scale*torch.sigmoid(raw_pred[...,0:2]), object_mask * raw_true_xy)/ m
            wh_loss = torch.sum(object_mask * box_loss_scale * 0.5 * (raw_true_wh-raw_pred[...,2:4])**2)/m

            confidence_loss = (self.mse_loss(torch.sigmoid(raw_pred[...,4:5])[object_mask == 1], object_mask[object_mask==1]) + \
                            self.mse_loss(torch.sigmoid(raw_pred[...,4:5])[((1-object_mask)*ignore_mask) == 1], object_mask[((1-object_mask)*ignore_mask) == 1]))/m

            class_loss = self.mse_loss(torch.sigmoid(raw_pred[...,5:])* object_mask, true_class_probs * object_mask)/m

            loss_xy += xy_loss
            loss_wh += wh_loss
            loss_conf += confidence_loss
            loss_clss += class_loss
            # loss += xy_loss + wh_loss + confidence_loss + class_loss
        
            nRecall += torch.sum(best_ious > 0.5)
            nRecall75 += torch.sum(best_ious > 0.75)
            nProposal += torch.sum(torch.sigmoid(raw_pred[...,4:5]) > 0.25)

        loss = loss_xy + loss_wh + loss_conf + loss_clss
        # print('loss %.3f, xy %.3f, wh %.3f, conf %.3f, class_loss: %.3f, nRecall: %d, nRecall75: %d, nProposal: %d' \
                # %(loss.item(), xy_loss.item(), wh_loss.item(), confidence_loss.item(), class_loss.item(), nRecall.item(), nRecall75.item(), nProposal.item()))

        return loss.unsqueeze(0), loss_xy.unsqueeze(0), loss_wh.unsqueeze(0), loss_conf.unsqueeze(0), \
                loss_clss.unsqueeze(0), nRecall.unsqueeze(0), nRecall75.unsqueeze(0), nProposal.unsqueeze(0)