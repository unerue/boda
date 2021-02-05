from boda.models import YolactConfig, YolactModel
# from boda.lib.torchinfo import summary
from boda.lib.torchsummary import summary


config = YolactConfig(num_classes=80)
model = YolactModel(config).to('cuda')
model.train()
print(model)
# print(summary(model, input_size=(16, 3, 550, 550), verbose=0))
print(summary(model, input_data=(3, 550, 550), verbose=0))
