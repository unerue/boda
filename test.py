from boda.architecture_yolov1 import Yolov1Model
from boda.configuration_yolov1 import Yolov1Config
from torchsummary import summary

import torch

config = Yolov1Config()
print(config)

model = Yolov1Model(config).to('cuda')
# print(summary(model, input_data=(3, 448, 448), verbose=0))
print(summary(model, input_data=(3, 448, 448), verbose=0))
