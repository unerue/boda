from boda.configuration_ssd import SsdConfig
from boda.backbone_vgg import vgg16
from boda.architecture_ssd import SsdModel
from torchsummary import summary


model = vgg16('test').to('cuda')
print(summary(model, input_data=(3, 300, 300), verbose=0))

config = SsdConfig()
model = SsdModel(config).to('cuda')
print(summary(model, input_data=(3, 300, 300), verbose=0))