from typing import Sequence

from torch import nn, Tensor

from . import backbone_resnet
from . import neck_fpn


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, fpn: nn.Module, **kwargs):
        super().__init__()
        
        self.backbone = backbone
        self.fpn = fpn

    def forward(self, x: Tensor) -> Sequence[Tensor]:
        x = self.backbone(x)
        x = self.fpn(x)
        return x


def resnet_fpn_extractor(backbone_name: str, fpn_name: str, fpn_channels: int = 256):
    backbone = backbone_resnet.__dict__[backbone_name]()
    backbone_channels = backbone.channels
    fpn = neck_fpn.__dict__[fpn_name](channels=backbone_channels, out_channels=fpn_channels)
    
    return FeatureExtractor(backbone, fpn)
