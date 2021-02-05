import math
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn, Tensor

from typing import List, Optional, Dict
from .image_list import ImageList

import numpy as np
import torch
from torch import nn


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Args:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.
    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.
    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.
    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.
    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(self, scales: List[int], aspect_ratios: List[float], dtype: torch.dtype = torch.float32,
                         device: torch.device = torch.device("cpu")):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(
                sizes,
                aspect_ratios,
                dtype,
                device
            )
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError("Anchors should be Tuple[Tuple[int]] because each feature "
                             "map could potentially have different sizes and aspect ratios. "
                             "There needs to be a match between the number of "
                             "feature maps passed and the number of sizes / aspect ratios specified.")

        for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors
        ):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def cached_grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for i in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors










# class BufferList(nn.Module):
#     """
#     Similar to nn.ParameterList, but for buffers
#     """

#     def __init__(self, buffers=None):
#         super(BufferList, self).__init__()
#         if buffers is not None:
#             self.extend(buffers)

#     def extend(self, buffers):
#         offset = len(self)
#         for i, buffer in enumerate(buffers):
#             self.register_buffer(str(offset + i), buffer)
#         return self

#     def __len__(self):
#         return len(self._buffers)

#     def __iter__(self):
#         return iter(self._buffers.values())


# class AnchorGenerator(nn.Module):
#     """
#     For a set of image sizes and feature maps, computes a set
#     of anchors
#     """

#     def __init__(
#         self,
#         sizes=(128, 256, 512),
#         aspect_ratios=(0.5, 1.0, 2.0),
#         anchor_strides=(8, 16, 32),
#         straddle_thresh=0,
#     ) -> None:
#         super().__init__()

#         if len(anchor_strides) == 1:
#             anchor_stride = anchor_strides[0]
#             cell_anchors = [
#                 generate_anchors(anchor_stride, sizes, aspect_ratios).float()
#             ]
#         else:
#             if len(anchor_strides) != len(sizes):
#                 raise RuntimeError("FPN should have #anchor_strides == #sizes")

#             cell_anchors = [
#                 generate_anchors(
#                     anchor_stride,
#                     size if isinstance(size, (tuple, list)) else (size,),
#                     aspect_ratios
#                 ).float()
#                 for anchor_stride, size in zip(anchor_strides, sizes)
#             ]
#         self.strides = anchor_strides
#         self.cell_anchors = BufferList(cell_anchors)
#         self.straddle_thresh = straddle_thresh

#     def num_anchors_per_location(self):
#         return [len(cell_anchors) for cell_anchors in self.cell_anchors]

#     def grid_anchors(self, grid_sizes):
#         anchors = []
#         for size, stride, base_anchors in zip(
#             grid_sizes, self.strides, self.cell_anchors
#         ):
#             grid_height, grid_width = size
#             device = base_anchors.device
#             shifts_x = torch.arange(
#                 0, grid_width * stride, step=stride, dtype=torch.float32, device=device
#             )
#             shifts_y = torch.arange(
#                 0, grid_height * stride, step=stride, dtype=torch.float32, device=device
#             )
#             shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#             shift_x = shift_x.reshape(-1)
#             shift_y = shift_y.reshape(-1)
#             shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

#             anchors.append(
#                 (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
#             )

#         return anchors

#     def add_visibility_to(self, boxlist):
#         image_width, image_height = boxlist.size
#         anchors = boxlist.bbox
#         if self.straddle_thresh >= 0:
#             inds_inside = (
#                 (anchors[..., 0] >= -self.straddle_thresh)
#                 & (anchors[..., 1] >= -self.straddle_thresh)
#                 & (anchors[..., 2] < image_width + self.straddle_thresh)
#                 & (anchors[..., 3] < image_height + self.straddle_thresh)
#             )
#         else:
#             device = anchors.device
#             inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)

#         boxlist.add_field("visibility", inds_inside)

#     def forward(self, image_list, feature_maps):
#         grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
#         anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
#         anchors = []
#         for i, (image_height, image_width) in enumerate(image_list.image_sizes):
#             anchors_in_image = []
#             for anchors_per_feature_map in anchors_over_all_feature_maps:
#                 boxlist = BoxList(
#                     anchors_per_feature_map, (image_width, image_height), mode="xyxy"
#                 )
#                 self.add_visibility_to(boxlist)
#                 anchors_in_image.append(boxlist)
#             anchors.append(anchors_in_image)
#         return anchors


# def make_anchor_generator(config):
#     anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
#     aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
#     anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
#     straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

#     if config.MODEL.RPN.USE_FPN:
#         assert len(anchor_stride) == len(
#             anchor_sizes
#         ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
#     else:
#         assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
#     anchor_generator = AnchorGenerator(
#         anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh
#     )
#     return anchor_generator


# def make_anchor_generator_retinanet(config):
#     anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
#     aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
#     anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
#     straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
#     octave = config.MODEL.RETINANET.OCTAVE
#     scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE

#     assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
#     new_anchor_sizes = []
#     for size in anchor_sizes:
#         per_layer_anchor_sizes = []
#         for scale_per_octave in range(scales_per_octave):
#             octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
#             per_layer_anchor_sizes.append(octave_scale * size)
#         new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

#     anchor_generator = AnchorGenerator(
#         tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
#     )
#     return anchor_generator

# # Copyright (c) 2017-present, Facebook, Inc.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# ##############################################################################
# #
# # Based on:
# # --------------------------------------------------------
# # Faster R-CNN
# # Copyright (c) 2015 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Ross Girshick and Sean Bell
# # --------------------------------------------------------


# # Verify that we compute the same anchors as Shaoqing's matlab implementation:
# #
# #    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
# #    >> anchors
# #
# #    anchors =
# #
# #       -83   -39   100    56
# #      -175   -87   192   104
# #      -359  -183   376   200
# #       -55   -55    72    72
# #      -119  -119   136   136
# #      -247  -247   264   264
# #       -35   -79    52    96
# #       -79  -167    96   184
# #      -167  -343   184   360

# # array([[ -83.,  -39.,  100.,   56.],
# #        [-175.,  -87.,  192.,  104.],
# #        [-359., -183.,  376.,  200.],
# #        [ -55.,  -55.,   72.,   72.],
# #        [-119., -119.,  136.,  136.],
# #        [-247., -247.,  264.,  264.],
# #        [ -35.,  -79.,   52.,   96.],
# #        [ -79., -167.,   96.,  184.],
# #        [-167., -343.,  184.,  360.]])


# def generate_anchors(
#     stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
# ):
#     """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
#     are centered on stride / 2, have (approximate) sqrt areas of the specified
#     sizes, and aspect ratios as given.
#     """
#     return _generate_anchors(
#         stride,
#         np.array(sizes, dtype=np.float) / stride,
#         np.array(aspect_ratios, dtype=np.float),
#     )


# def _generate_anchors(base_size, scales, aspect_ratios):
#     """Generate anchor (reference) windows by enumerating aspect ratios X
#     scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
#     """
#     anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
#     anchors = _ratio_enum(anchor, aspect_ratios)
#     anchors = np.vstack(
#         [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
#     )
#     return torch.from_numpy(anchors)


# def _whctrs(anchor):
#     """Return width, height, x center, and y center for an anchor (window)."""
#     w = anchor[2] - anchor[0] + 1
#     h = anchor[3] - anchor[1] + 1
#     x_ctr = anchor[0] + 0.5 * (w - 1)
#     y_ctr = anchor[1] + 0.5 * (h - 1)
#     return w, h, x_ctr, y_ctr


# def _mkanchors(ws, hs, x_ctr, y_ctr):
#     """Given a vector of widths (ws) and heights (hs) around a center
#     (x_ctr, y_ctr), output a set of anchors (windows).
#     """
#     ws = ws[:, np.newaxis]
#     hs = hs[:, np.newaxis]
#     anchors = np.hstack(
#         (
#             x_ctr - 0.5 * (ws - 1),
#             y_ctr - 0.5 * (hs - 1),
#             x_ctr + 0.5 * (ws - 1),
#             y_ctr + 0.5 * (hs - 1),
#         )
#     )
#     return anchors


# def _ratio_enum(anchor, ratios):
#     """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
#     w, h, x_ctr, y_ctr = _whctrs(anchor)
#     size = w * h
#     size_ratios = size / ratios
#     ws = np.round(np.sqrt(size_ratios))
#     hs = np.round(ws * ratios)
#     anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
#     return anchors


# def _scale_enum(anchor, scales):
#     """Enumerate a set of anchors for each scale wrt an anchor."""
#     w, h, x_ctr, y_ctr = _whctrs(anchor)
#     ws = w * scales
#     hs = h * scales
#     anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
#     return anchors
