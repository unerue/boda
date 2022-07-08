from boda.models import SsdModel, SsdConfig


from boda.lib.torchsummary import summary
import torch

config = SsdConfig(num_classes=80)
model = SsdModel(config).to('cuda')
model.eval()
print(model)
# print(summary(model, input_size=(16, 3, 550, 550), verbose=0))
print(summary(model, input_data=(3, 550, 550), verbose=0))