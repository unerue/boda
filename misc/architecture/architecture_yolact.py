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

class Yolact:
    def __init__(self, backbone):
        self.backbone = backbone
        self.fpn = FPN([512, 1024, 2048])
        backbone_outs=[]
        for i in [1,2,3]:
            backbone_outs.append(self.backbone[i])

        self.fpn_outs = self.fpn(backbone_outs)
    

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




