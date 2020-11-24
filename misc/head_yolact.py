import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Networks (FPN)
    https://arxiv.org/pdf/1612.03144.pdf
    """
    def __init__(self, in_channels=[512, 1024, 2048]):
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
                self.fpn_num_features, self.fpn_num_features, 
                kernel_size=3, padding=self.fpn_pad) for _ in in_channels])

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

    def forward(self, convouts):
        """
        backbone_outs = [[n, 512, 69, 69], [n, 1024, 35, 35], [n, 2048, 18, 18]]
        In class Yolact's train(), remove C2 from bakebone_outs. So FPN gets three feature outs.
        """
        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        j = len(convouts)
        for lateral_layer in self.lateral_layers:
            j -= 1
            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolate_mode, align_corners=False)
            
            x = x + lateral_layer(convouts[j])
            out[j] = x

        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        for downsample_layer in self.downsample_layers:
            out.append(downsample_layer(out[-1]))

        return out



if __name__ == '__main__':
    from torchsummary import summary
    from backbone.backbone_yolact import construct_backbone

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = construct_backbone().to(device)
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




mask_proto_net = [
    (256, 3, {'padding': 1}), 
    (256, 3, {'padding': 1}), 
    (256, 3, {'padding': 1}),
    (None, -2, {}), 
    (256, 3, {'padding': 1}), 
    (6, 1, {})  # (32, 1, {})
]

class Protonet(nn.Module) :
    def __init__(self, mask_proto_net) :
        super().__init__()

        self.inplanes = 256
        self.mask_proto_net = mask_proto_net
        self.conv1 = nn.Conv2d(self.inplanes, mask_proto_net[0][0], kernel_size=mask_proto_net[0][1], **mask_proto_net[0][2])
        self.conv2 = nn.Conv2d(self.inplanes, mask_proto_net[1][0], kernel_size=mask_proto_net[1][1], **mask_proto_net[1][2])
        self.conv3 = nn.Conv2d(self.inplanes, mask_proto_net[2][0], kernel_size=mask_proto_net[2][1], **mask_proto_net[2][2])
        self.conv4 = nn.Conv2d(self.inplanes, mask_proto_net[4][0], kernel_size=mask_proto_net[4][1], **mask_proto_net[4][2])
        self.conv5 = nn.Conv2d(self.inplanes, mask_proto_net[5][0], kernel_size=mask_proto_net[5][1], **mask_proto_net[5][2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor=-self.mask_proto_net[3][1], mode='bilinear', align_corners=False, **self.mask_proto_net[3][2])
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        
        return out

proto_out = Protonet(mask_proto_net)(fpn_outs[0])
print(f'Proto net shape: {proto_out.size()}')

# class YolactModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.backbone = None





# class YolactPredictionHead(nn.Module):
#     def __init__(
#         self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
#         super().__init__()

#         self.num_classes = None
#         self.mask_dim = None
#         self.num_priors = sum(len(x) * len(scales) for x in aspect_ratios)
#         self.parent = [parent]
#         self.index = index
#         self.num_heads = None

        


#         if parent is None:
            
#             self.bbox_layer = nn.Conv2d()
#             self.conf_layer = nn.Conv2d()
#             self.mask_layer = nn.Conv2d()


#     def forward(self, x):
#         conv_h = x.size(2)
#         conv_w = x.size(3)

    
#     def make_priors(self, conv_h, conv_w, device):
#         pass


