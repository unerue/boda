from boda.models.architecture_yolov1 import Yolov1Model
from boda.models.configuration_yolov1 import Yolov1Config
from boda.models.loss_yolov1 import Yolov1Loss
from boda.models.backbone_darknet import darknet, darknet21
from torchsummary import summary

import torch

# model = darknet()
# print(summary(model, input_data=(3, 448, 448), verbose=0))

# model = darknet21()
# print(summary(model, input_data=(3, 448, 448), verbose=0))


config = Yolov1Config()
# print(config)

# model = Yolov1Model(config).to('cuda')
# print(summary(model, input_data=(3, 448, 448), verbose=0))


# train_loader = [[torch.rand((3, 448, 448)).to('cuda') for _ in range(16)]]
# print(len(train_loader))
# print(torch.rand((3, 448, 448)).size())
# print(torch.rand((3, 448, 448)).dim())

# num_epochs = 10
# for epoch in range(num_epochs):
#     for inputs in train_loader:
#         print(epoch)
#         model(inputs)

# print(summary(model, input_data=(3, 448, 448), verbose=0))



criterion = Yolov1Loss(config)


targets = [{
    'boxes': torch.tensor(
        [[120.1, 200.1, 150.1, 250.1], [0.1, 50.1, 100.1, 300.1]]).to('cuda'),
    'labels': torch.tensor([1, 2]).to('cuda'),
}, {'boxes': torch.tensor(
        [[80.1, 200.1, 150.1, 250.1], [20, 10, 40, 50], [5, 50.1, 150.1, 400.1]]).to('cuda'),
    'labels': torch.tensor([1, 2, 10]).to('cuda')}]

# preds = {
#     'boxes': torch.tensor(
#         [[[0.08, 0.03, 0.21, 0.55],
#           [0.01, 0.02, 0.22, 0.55], 
#           [0.5, 0.7, 0.50, 0.80], 
#           [0.95, 0.5, 0.1, 0.1]], 
#          [[0.01, 0.02, 0.22, 0.55], 
#           [0.08, 0.03, 0.21, 0.55],
#           [0.5, 0.7, 0.50, 0.80], 
#           [0.95, 0.5, 0.1, 0.1]]]).to('cuda'),
#     'scores': torch.tensor([
#         [0.5, 0.1, 0.05, 0.6],
#         [0.01, 0.05, 0.05, 0.7]
#     ]).to('cuda')
# }

inputs = [torch.randn((3, 448, 448)).to('cuda') for _ in range(2)]
model = Yolov1Model(config).to('cuda')
preds = model(inputs)
losses = criterion(preds, targets)
print(losses)