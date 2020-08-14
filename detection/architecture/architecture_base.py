from typing import List
from torch import Tensor


def _check_inputs(inputs: List[Tensor]):
    for image in inputs:
        if isinstance(image, Tensor):
            if image.dim() != 3:
                raise ValueError(f'images is expected to be 3d tensors of shape [C, H, W] {image.shape}')
    
        else:
            raise ValueError('Expected image to be Tensor.')