from boda.configuration_yolact import YolactConfig
from boda.backbone_resnet import resnet101
from boda.architecture_yolact import YolactModel
from torchsummary import summary


model = resnet101().to('cuda')
print(summary(model, input_data=(3, 550, 550), verbose=0))

config = YolactConfig()
model = YolactModel(config).to('cuda')
print(summary(model, input_data=(3, 550, 550), verbose=0))