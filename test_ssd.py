import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from boda.models.configuration_ssd import SsdConfig
from boda.models.backbone_vgg import vgg16
from boda.models.architecture_ssd import SsdModel
# from boda.models.loss_ssd import SsdLoss

from torchsummary import summary



model = vgg16()
print(summary(model, input_data=(3, 300, 300), verbose=0))

config = SsdConfig()
model = SsdModel(config)
print(summary(model, input_data=(3, 300, 300), verbose=0))


