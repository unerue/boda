from boda.configuration_yolact import YolactConfig
from boda.backbone_resnet import resnet101
from boda.architecture_yolact import YolactModel
from torchsummary import summary
import torch

model = resnet101().to('cuda')
# print(summary(model, input_data=(3, 550, 550), verbose=0))

config = YolactConfig()
model = YolactModel(config).to('cuda')
torch.save(model.state_dict(), 'yolact.pth')
# print(summary(model, input_data=(3, 550, 550), verbose=0))

state_dict = torch.load('yolact.pth')
for key, value in state_dict.items():
    if key[:8] != 'backbone':
        print(key, value.size())