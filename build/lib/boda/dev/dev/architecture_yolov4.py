import sys
from typing import Tuple, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

from ..modules import Mish


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size
        x.size()

        if self.training:
            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class ConvBnActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str = 'mish',
        norm_layer=nn.BatchNorm2d,
        bias=False
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=bias))

        if norm_layer is not None:
            self.layers.append(norm_layer(out_channels))

        if activation == 'mish':
            self.layers.append(Mish())
        elif activation == 'relu':
            self.layers.append(nn.ReLU())
        elif activation == 'reaky_relu':
            self.layers.append(nn.LeakyReLU(0.1))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs


class ResidualBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, num_blocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(num_blocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBnActivation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(ConvBnActivation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x  # TODO: ?!!!!
            for res in module:
                h = res(h)

            if self.shortcut:
                x = x + h
            else:
                x = h
            # x = x + h if self.shortcut else h
        return x


class Yolov4Neck(nn.Module):
    def __init__(self, channels: List, inference=False):
        super().__init__()
        # self.inference = inference
        # self.lateral_layers = nn.ModuleList([
        #     nn.Conv2d(
        #         _in_channels,
        #         config.fpn_out_channels,
        #         kernel_size=1) for _in_channels in reversed(channels)]
        self.layer1 = nn.Sequential(
            ConvBnActivation(1024, 512, 1, 1, 'leaky_relu'),
            ConvBnActivation(512, 1024, 3, 1, 'leaky_relu'),
            ConvBnActivation(1024, 512, 1, 1, 'leaky_relu')
        )
        # self.conv1 = ConvBnActivation(1024, 512, 1, 1, 'leaky')
        # self.conv2 = ConvBnActivation(512, 1024, 3, 1, 'leaky')
        # self.conv3 = ConvBnActivation(1024, 512, 1, 1, 'leaky')
        # SPP

        self.maxpools = nn.ModuleList()
        kernel_sizes = [5, 9, 13]
        for kernel_size in kernel_sizes:
            self.maxpools.append(
                nn.MaxPool2d(
                    kenel_size=kernel_size, stride=1, padding=kernel_size // 2))
        # self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.layer2 = nn.Sequential(
            ConvBnActivation(2048, 512, 1, 1, 'leaky_relu'),
            ConvBnActivation(512, 1024, 3, 1, 'leaky'),
            ConvBnActivation(1024, 512, 1, 1, 'leaky'),
            ConvBnActivation(512, 256, 1, 1, 'leaky')
        )
        # self.conv4 = ConvBnActivation(2048, 512, 1, 1, 'leaky')
        # self.conv5 = ConvBnActivation(512, 1024, 3, 1, 'leaky')
        # self.conv6 = ConvBnActivation(1024, 512, 1, 1, 'leaky')
        # self.conv7 = ConvBnActivation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = ConvBnActivation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = ConvBnActivation(512, 256, 1, 1, 'leaky')
        self.conv10 = ConvBnActivation(256, 512, 3, 1, 'leaky')
        self.conv11 = ConvBnActivation(512, 256, 1, 1, 'leaky')
        self.conv12 = ConvBnActivation(256, 512, 3, 1, 'leaky')
        self.conv13 = ConvBnActivation(512, 256, 1, 1, 'leaky')
        self.conv14 = ConvBnActivation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = ConvBnActivation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = ConvBnActivation(256, 128, 1, 1, 'leaky')
        self.conv17 = ConvBnActivation(128, 256, 3, 1, 'leaky')
        self.conv18 = ConvBnActivation(256, 128, 1, 1, 'leaky')
        self.conv19 = ConvBnActivation(128, 256, 3, 1, 'leaky')
        self.conv20 = ConvBnActivation(256, 128, 1, 1, 'leaky')

    def forward(self, inputs, downsample4, downsample3, inference=False):
        # x1 = self.conv1(input)
        # x2 = self.conv2(x1)
        # x3 = self.conv3(x2)
        for layer in self.layer1:
            inputs = layer(inputs)
        
        outputs = []
        for maxpool in self.maxpools:
            outputs.append(maxpool(inputs))

        outputs = torch.cat(outputs, dim=1)
        # # SPP
        # m1 = self.maxpool1(x3)
        # m2 = self.maxpool2(x3)
        # m3 = self.maxpool3(x3)
        # spp = torch.cat([m3, m2, m1, x3], dim=1)

        # SPP end
        for layer in self.layer2:
            outputs = layer(outputs)
        # x4 = self.conv4(spp)
        # x5 = self.conv5(x4)
        # x6 = self.conv6(x5)
        # x7 = self.conv7(x6)

        # Up
        outputs = self.upsample1(outputs, downsample4, self.training)
        # up = self.upsample1(x7, downsample4.size(), self.inference)

        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


def yolo_forward(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0), axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii : ii + 1] + torch.tensor(grid_x, device=device, dtype=torch.float32) # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(grid_y, device=device, dtype=torch.float32) # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)


    ########################################
    #   Figure out bboxes from slices     #
    ########################################
    
    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= W
    by_bh /= H

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(batch, num_anchors * H * W, 1)
    by = by_bh[:, :num_anchors].view(batch, num_anchors * H * W, 1)
    bw = bx_bw[:, num_anchors:].view(batch, num_anchors * H * W, 1)
    bh = by_bh[:, num_anchors:].view(batch, num_anchors * H * W, 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(batch, num_anchors * H * W, 1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(batch, num_anchors * H * W, 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return  boxes, confs


def yolo_forward_dynamic(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    # batch = output.size(0)
    # H = output.size(2)
    # W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3))

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(output.size(0), num_anchors, num_classes, output.size(2) * output.size(3))
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(output.size(0), num_anchors * output.size(2) * output.size(3), num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0), axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii : ii + 1] + torch.tensor(grid_x, device=device, dtype=torch.float32) # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(grid_y, device=device, dtype=torch.float32) # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)


    ########################################
    #   Figure out bboxes from slices     #
    ########################################
    
    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    by = by_bh[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bw = bx_bw[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bh = by_bh[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return boxes, confs


class HeadBranch(nn.Module):
    def __init__(
        self,
        anchor_mask: List = [],
        num_classes: int = 0,
        anchors: List = [],
        num_anchors: int = 1,
        stride: int = 32,
        model_out: bool = False
    ) -> None:
        super().__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            return output

        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return yolo_forward_dynamic(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),scale_x_y=self.scale_x_y)


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = ConvBnActivation(128, 256, 3, 1, 'leaky')
        self.conv2 = ConvBnActivation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                                anchor_mask=[0, 1, 2], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)

        # R -4
        self.conv3 = ConvBnActivation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = ConvBnActivation(512, 256, 1, 1, 'leaky')
        self.conv5 = ConvBnActivation(256, 512, 3, 1, 'leaky')
        self.conv6 = ConvBnActivation(512, 256, 1, 1, 'leaky')
        self.conv7 = ConvBnActivation(256, 512, 3, 1, 'leaky')
        self.conv8 = ConvBnActivation(512, 256, 1, 1, 'leaky')
        self.conv9 = ConvBnActivation(256, 512, 3, 1, 'leaky')
        self.conv10 = ConvBnActivation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)

        # R -4
        self.conv11 = ConvBnActivation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = ConvBnActivation(1024, 512, 1, 1, 'leaky')
        self.conv13 = ConvBnActivation(512, 1024, 3, 1, 'leaky')
        self.conv14 = ConvBnActivation(1024, 512, 1, 1, 'leaky')
        self.conv15 = ConvBnActivation(512, 1024, 3, 1, 'leaky')
        self.conv16 = ConvBnActivation(1024, 512, 1, 1, 'leaky')
        self.conv17 = ConvBnActivation(512, 1024, 3, 1, 'leaky')
        self.conv18 = ConvBnActivation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])
        
        else:
            return [x2, x10, x18]


class Yolov4Model(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        # self.down1 = DownSample1()
        # self.down2 = DownSample2()
        # self.down3 = DownSample3()
        # self.down4 = DownSample4()
        # self.down5 = DownSample5()
        self.backbone = resnet50()
        # neck
        self.neek = Neck(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neek)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        
        # head
        self.head = Yolov4Head(output_ch, n_classes, inference)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output
