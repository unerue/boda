from boda.architecture_yolov1 import Yolov1Model
from boda.configuration_yolov1 import Yolov1Config
from boda.loss_yolov1 import Yolov1Loss
from torchsummary import summary

import torch

config = Yolov1Config()
print(config)

# model = Yolov1Model(config).to('cuda')
# print(summary(model, input_data=(3, 448, 448), verbose=0))
# print(summary(model, input_data=(3, 448, 448), verbose=0))



criterion = Yolov1Loss(config)

targets = [{
    'boxes': torch.tensor(
        [[100.1, 200.1, 150.1, 250.1], [0.1, 50.1, 100.1, 300.1]]),
    'labels': torch.tensor([1, 2]),
}, {'boxes': torch.tensor(
        [[100.1, 200.1, 150.1, 250.1], [20, 30, 40, 50], [0.1, 50.1, 100.1, 300.1]]),
    'labels': torch.tensor([1, 2])}]

preds = {
    'boxes': torch.tensor(
        [[[0.08, 0.03, 0.21, 0.55],
          [0.01, 0.02, 0.22, 0.55], 
          [0.5, 0.7, 0.50, 0.80], 
          [0.95, 0.5, 0.1, 0.1]], 
         [[0.01, 0.02, 0.22, 0.55], 
          [0.08, 0.03, 0.21, 0.55],
          [0.5, 0.7, 0.50, 0.80], 
          [0.95, 0.5, 0.1, 0.1]]]),
    'scores': torch.tensor([
        [0.5, 0.1, 0.05, 0.6],
        [0.01, 0.05, 0.05, 0.7]
    ])
}

criterion(preds, targets)