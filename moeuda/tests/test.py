import torch
from torchsummary import summary

from backbone import YolactBackbone
from architecture import Yolact


backbone_config = {
    'name': 'resnet50', 
    'path': 'resnet50-19c8e357.pth',
    'selected_layers': list(range(1, 4)), 
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
backbone = YolactBackbone().to(device)
print(summary(backbone, input_data=(3, 550, 550), verbose=0))

input_data = torch.randn(1, 3, 550, 550)
backbone = YolactBackbone()(input_data)
print(f'C2: {backbone[0].size()}')
print(f'C3: {backbone[1].size()}')
print(f'C4: {backbone[2].size()}')
print(f'C5: {backbone[3].size()}')

yolact = Yolact(backbone)
print(f'P3 shape: {yolact.fpn_outs[0].size()}')
print(f'P4 shape: {yolact.fpn_outs[1].size()}')
print(f'P5 shape: {yolact.fpn_outs[2].size()}')
print(f'P6 shape: {yolact.fpn_outs[3].size()}')
print(f'P7 shape: {yolact.fpn_outs[4].size()}')
