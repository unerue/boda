import torch.nn as nn


class Yolov1Model(Backbone):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = Backbone.get_backbone()
        self.num_grid = None
        self.last_conv = nn.Sequential([])
        self.num_class = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.last_conv(x)
        x = x.view(x.size(0), -1)
        