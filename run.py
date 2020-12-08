from boda import YolactConfig, YolactModel
from boda.lib.torchsummary import summary
import torch

config = YolactConfig()
# print(config)
# model = YolactModel.from_pretrained('yolact')
# for k, v in model.items():
#     print(k, v)
# print(summary(model, input_data=(3, 550, 550), verbose=0))


model = YolactModel(config)
print(summary(model, input_data=(3, 550, 550), verbose=0))
torch.save(model.state_dict(), 'yolact.pth')
