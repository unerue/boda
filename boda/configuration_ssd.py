from .configuration_base import BaseConfig


SSD_PRETRAINED_CONFIG = {
    'ssd300': None,
    'ssd500': None,
}


class SsdConfig(BaseConfig):
    """Configuration for SSD

    Arguments:
        max_size ():

    """
    def __init__(
        self, 
        selected_layers=-1,
        grid_size=7, 
        max_size=300,
        num_classes=20,
        **kwargs):
        super().__init__(max_size=max_size, **kwargs)
        self.selected_layers = selected_layers
        self.boxes = [4, 6, 6, 6, 4, 4]
        self.num_classes = num_classes
        self.backbone_name = 'vgg16'
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.variance = [0.1, 0.2]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.clip = True
        
        # self.grid_sizes = [38, 19, 10, 5, 3, 1]


