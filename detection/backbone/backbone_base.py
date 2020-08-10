from .resnet import resnet_50


resnet = {
    'resnet_50': resnet_50(),
    'resnet_101': resnet_101()
}


class Backbone:
    def __init__(self, config):
        pass

    def get_backbone(self):
        pass