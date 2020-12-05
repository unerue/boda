import torch


state_dict = torch.load('resnet50_550_coco_972_70000.pth')
print(state_dict.keys())

for key in state_dict.keys():
    if key[:8] != 'backbone':
        print(key)