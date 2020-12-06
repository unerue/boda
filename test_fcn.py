# from boda.configuration_ssd import SsdConfig
from boda.backbone_vgg import vgg16
from boda.architecture_fcn import FcnModel
from torchsummary import summary


model = vgg16('test').to('cuda')
print(summary(model, input_data=(3, 224, 224), verbose=0))

# config = SsdConfig()
model = FcnModel({}).to('cuda')
print(summary(model, input_data=(3, 224, 224), verbose=0))
