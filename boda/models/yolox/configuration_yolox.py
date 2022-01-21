from typing import List

from ...base_configuration import BaseConfig


class YoloXConfig(BaseConfig):
    model_name = 'yolox'
    
    def __init__(
        self,
        num_classes: int = 80,
        image_size: int = 640,
        depth: float = 1.0,
        width: float = 1.0,
        act: str = 'silu',
        selected_backbone_layers: List[int] = [2, 3, 4],
        depthwise: bool = False,
        test_conf: float = 0.01,
        nmsthre: float = 0.65,
    ):
        super().__init__(
            num_classes=num_classes,
            max_size=image_size,
        )
        self.depth = depth
        self.width = width
        self.act = act
        
        self.selected_backbone_layers = selected_backbone_layers
        
        self.depthwise = depthwise
        
        self.test_conf = test_conf
        self.nmsthre = nmsthre
