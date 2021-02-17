# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
from typing import List, Tuple, Dict, Optional
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from .roi_heads import paste_masks_in_image


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

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


def _resize_image_and_masks(
    image: Tensor,
    self_min_size: float,
    self_max_size: float,
    target: Optional[Dict[str, Tensor]]
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    scale_factor = self_min_size / min_size
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size

    image = torch.nn.functional.interpolate(
        image[None],
        scale_factor=scale_factor,
        mode='bilinear',
        recompute_scale_factor=True,
        align_corners=False
    )[0]

    if target is None:
        return image, target

    if 'masks' in target:
        mask = target['masks']
        mask = F.interpolate(
            mask[:, None].float(),
            scale_factor=scale_factor,
            recompute_scale_factor=True
        )[:, 0].byte()
        target['masks'] = mask

    return image, target


class RcnnTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """
    def __init__(self, min_size: int, max_size: int, image_mean: List[float], image_std: List[float]):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        if targets is not None:
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy

        images = [image for image in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(
                    'images is expected to be a list of 3d tensors '
                    f'of shape [C, H, W], got {image.shape}'
                )
            # image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [image.shape[-2:] for image in images]
        images = self.batch_images(images)
        # TODO: duplicated??
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return image_list, targets

    def normalize(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f'Expected input images to be of floating type (in range [0, 1]), '
                f'but found type {image.dtype} instead'
            )

        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image: Tensor, target: Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])

        image, target = _resize_image_and_masks(image, size, float(self.max_size), target)

        if target is None:
            return image, target

        bbox = target['boxes']
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target['boxes'] = bbox

        if 'keypoints' in target:
            keypoints = target['keypoints']
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target['keypoints'] = keypoints

        return image, target

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)

        return maxes

    def batch_images(self, images: List[Tensor], size_divisible: int = 32):
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]]
    )-> List[Dict[str, Tensor]]:
        if self.training:
            return result

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred['boxes']
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes

            if 'masks' in pred:
                masks = pred['masks']
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]['masks'] = masks

            if 'keypoints' in pred:
                keypoints = pred['keypoints']
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]['keypoints'] = keypoints

        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        # format_string += f'{_indent}Normalize(mean={self.image_mean}, std={self.image_std})'
        format_string += f'{_indent}Resize(min_size={self.min_size}, max_size={self.max_size}, mode=bilinear)'
        format_string += '\n)'
        return format_string


def resize_keypoints(keypoints: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=keypoints.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h

    return resized_data


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]

    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
