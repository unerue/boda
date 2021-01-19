import os
import sys
import torch

if __name__ == '__main__':
    pass

sys.path.append('../')
# pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(pytorch_test_dir)

from boda.models.configuration_yolact import YolactConfig
from boda.models.backbone_resnet import resnet101
from boda.models.architecture_yolact import YolactModel
from boda.lib.torchsummary import summary
# import torch

model = resnet101().to('cuda')
# print(summary(model, input_data=(3, 550, 550), verbose=0))

config = YolactConfig(use_jit=True)
print(config)
model = YolactModel(config).to('cuda')
torch.save(model.state_dict(), 'yolact.pth')
print(summary(model, input_data=(3, 550, 550), verbose=0))

state_dict = torch.load('yolact.pth')
for key, value in state_dict.items():
    if key[:8] != 'backbone':
        print(key, value.size())