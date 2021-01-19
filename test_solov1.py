import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from boda.utils.dataset import CocoDataset
# from boda.utils.transforms import Compose, Resize, ToTensor, Normalize
# from boda.models.configuration_solov1 import Solov1Config
# from boda.models.backbone_resnet import resnet101
# from boda.models.architecture_solov1 import Solov1Model, Solov1PredictHead, Solov1PredictNeck
# from boda.models.loss_solov1 import Solov1Loss
# from boda.utils.trainer import Trainer
from boda.lib.torchsummary import summary

from boda.models import Solov1Model, Solov1Config


# model = Solov1PredictNeck().to('cuda')
config = Solov1Config()
model = Solov1Model(config).to('cuda')
print(summary(model, input_data=(3, 1333, 800), verbose=0))
