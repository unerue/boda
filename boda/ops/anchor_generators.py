import functools
import itertools
import math
from collections import defaultdict
from typing import List, Tuple

import torch
from torch import Tensor


def default_box_cache(func):
    cache = defaultdict()

    @functools.wraps(func)
    def wrapper(*args):
        k, v = func(*args)
        if k not in cache:
            cache[k] = v
        return k, cache[k]

    return wrapper


class DefaultBoxGenerator:
    """
    Args:
        aspect_ratios (:obj:`List[int]`):
        scales (:obj:):
        max_size ():
        use_preapply_sqrt ():
        use_pixel_scales ():
        use_square_anchors (:obj:`bool`): default `True`
    """

    def __init__(
        self,
        aspect_ratios: List[int],
        scales: List[float],
        max_size: Tuple[int] = (550, 550),
        use_preapply_sqrt: bool = True,
        use_pixel_scales: bool = True,
        use_square_anchors: bool = True,
    ) -> None:
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.clip = False
        self.max_size = max_size
        self.use_preapply_sqrt = use_preapply_sqrt
        self.use_pixel_scales = use_pixel_scales
        self.use_square_anchors = use_square_anchors

    @default_box_cache
    def generate(
        self, h: int, w: int, device: str = "cuda:0"
    ) -> Tuple[Tuple[int], Tensor]:
        """DefaultBoxGenerator is

        Args:
            h (:obj:`int`): feature map size from backbone
            w (:obj:`int`): feature map size from backbone
            device (:obj:`str`): default `cuda`

        Returns
            size (:obj:`Tuple[int]`): feature map size
            prior_boxes (:obj:`FloatTensor[N, 4]`):
        """
        size = (h, w)
        default_boxes = []
        for j, i in itertools.product(range(h), range(w)):
            cx = (i + 0.5) / w
            cy = (j + 0.5) / h
            for ratios in self.aspect_ratios:
                for scale in self.scales:
                    for ratio in ratios:
                        if not self.use_preapply_sqrt:
                            ratio = math.sqrt(ratio)

                        if self.use_pixel_scales:
                            _h = scale / ratio / self.max_size[0]
                            _w = scale * ratio / self.max_size[1]
                        else:
                            _h = scale / ratio / h
                            _w = scale * ratio / w

                        if self.use_square_anchors:
                            _h = _w

                        default_boxes += [cx, cy, _w, _h]

        default_boxes = torch.tensor(
            default_boxes, dtype=torch.float32, device=device, requires_grad=False
        ).view(-1, 4)
        if self.clip:
            default_boxes.clamp_(min=0, max=1)
        # prior_boxes.requires_grad = False

        return size, default_boxes
