import torch
from boda.lib.torchsummary import summary
from boda.models import FasterRcnnConfig, FasterRcnnModel


config = FasterRcnnConfig()
model = FasterRcnnModel(config).to('cuda')
model.eval()
# print(summary(model, input_data=(3, 1333, 800), verbose=0))
outputs = model([torch.randn(3, 1333, 800).to('cuda')])
print(outputs['0'].size())
