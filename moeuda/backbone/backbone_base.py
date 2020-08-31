from typing import Tuple, List


class SelectBackbone:
    def __init__(self, config, backbone_list: List):
        self.config = config
        self.backbone_list = backbone_list
        
    def load_backbone(self):
        
        return