from torchsummary import summary
from boda.models import FasterRcnnConfig, FasterRcnnModel


config = FasterRcnnConfig()
model = FasterRcnnModel(config).to('cuda')
print(summary(model, input_data=(3, 1333, 800), verbose=0))
