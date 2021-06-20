from boda.models import YolactConfig, YolactModel
# from boda.lib.torchsummary import summary
from torchinfo import summary
from boda.models.backbone_mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from boda.models.backbone_resnet import resnet101, resnet18, resnet34, resnet50
# from torchvision.models import resnet101, mobilenet_v3_large


config = YolactConfig(num_classes=8)
# model = YolactModel(config, backbone=mobilenet_v3_small(), selected_backbone_layers=[3, 8, 11]).to('cuda:0')
model = YolactModel(config, backbone=mobilenet_v3_large(), selected_backbone_layers=[6, 12, 15]).to('cuda:0')
# model = YolactModel(config, backbone=resnet101(), selected_backbone_layers=[1, 2, 3]).to('cuda:0')
print(summary(model, (1, 3, 550, 550), verbose=0))

# from boda.resnet import resnet101
# model = mobilenet_v3_small().to('cuda')
# model = mobilenet_v3_large().to('cuda')
# print(summary(model, input_data=(3, 550, 550), depth=2, verbose=0))

# model = resnet101().to('cuda')
# print(summary(model, (1, 3, 550, 550), verbose=0))
