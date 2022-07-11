from ...base_configuration import BaseConfig


SSD_PRETRAINED_CONFIG = {
    "ssd300": None,
    "ssd512": None,
}


class SsdConfig(BaseConfig):
    """Configuration for SSD

    Arguments:
        max_size ():

    """

    def __init__(
        self,
        num_classes: int = 20,
        max_size: int = 300,
        preserve_aspect_ratio: bool = False,
        selected_layers: int = -1,
        num_grids: int = 7,
        **kwargs
    ) -> None:
        super().__init__(max_size=max_size, **kwargs)
        self.selected_layers = [3, 4]
        self.boxes = [4, 6, 6, 6, 4, 4]
        self.num_classes = num_classes
        self.backbone_name = "vgg16"
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.variance = [0.1, 0.2]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.clip = True
        # self.grid_sizes = [38, 19, 10, 5, 3, 1]
