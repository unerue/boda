import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPool2d(nn.Module):
    """Spatial Pyramid Pooling Layer

    Spatial Pyramid Pooling in Deep Convolutional Networks
    https://arxiv.org/abs/1406.4729

    Argument:
        pyramid_levels: an int list of expected output size of max pooling layer
    """
    def __init__(self, pyramid_levels: List[int] = [4, 2, 1]):
        self.pyramid_levels = pyramid_levels # output size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._spatial_pyramid_pooling(input, self.pyramid_levels)

    @staticmethod
    def _spatial_pyramid_pooling(input, pyramid_levels: List[int]):
        """Spatial Pyramid Pooling

        Arguments:
            input: feature_map: a tensor vector of previous convolution layer
            pyramid_levels: a int vector of expected output size of max pooling layer
        
        return:
            a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """
        # num_samples: an number of image in the batch
        num_samples = input.size(0)
        # feature_map_size: [height, width]
        feature_map_size = [int(input.size(2)), int(input.size(3))]
        for i, level in enumerate(pyramid_levels):
            h_wid = int(math.ceil(feature_map_size[0] / level))
            w_wid = int(math.ceil(feature_map_size[1] / level))

            h_pad = (h_wid*level - feature_map_size[0] + 1)/2
            w_pad = (w_wid*level - feature_map_size[1] + 1)/2

            max_pool = nn.MaxPool2d(
                (h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))

            spp = max_pool(input)
            if i == 0:
                out = spp.view(num_samples, -1)
            else:
                out = torch.cat(
                    (out, spp.view(num_samples, -1)), 1)

        return out

    
