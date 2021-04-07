import math
from typing import Tuple, List, Dict, Callable, Optional

import cv2
from numpy.core.fromnumeric import _resize_dispatcher, mean
import pycocotools
import torch
import torch.nn.functional as F
from torch import dtype, nn, Tensor
from torchvision import transforms
import numpy as np
from numpy import ndarray


# def _check_image(image: ndarray):
#     """Check Image Shape

#     Args:
#         image (:obj:`ndarray[C, H, W]`)
#     Return:
#         image (:obj:`ndarray[H, W, C]`)
#     """
#     if image.shape[0] == 3:
#         return image.transpose((1, 2, 0))


# class ResizeTargets:
#     def __init__(self) -> None:
#         pass


# class ResizeImages:
#     def __init__(
#         self,
#         size: Tuple[int],
#         min_size: int = 800,
#         max_size: int = 1333,
#         preserve_aspect_ratio: bool = False,
#         mode: str = 'bilinear'
#     ) -> None:
#         """
#         Args:
#             size (:obj:Tuple[int, int]):
#             min_size (:obj:int):
#             max_size (:obj:int):
#             preserve_aspect_ratio (:obj:bool):
#             mode (:obj:str): 
#         """
#         self.min_size = min_size
#         self.max_size = max_size
#         self.size = size
#         self.preserve_aspect_ratio = preserve_aspect_ratio
#         self.mode = mode

#     def __call__(
#         self,
#         images: List[Tensor],
#         targets: List[Dict[str, ndarray]]
#     ) -> Tuple[ndarray, Dict[str, ndarray]]:
#         """
#         Args:
#             image (:obj:`ndarray[C, H, W]`)
#             targets (:obj:`Dict[str, ndarray]`):
#                 `masks` (:obj:`ndarray[N, H, W]`):
#         """
#         for i in range(len(images)):
#             image = images[i]
#             target = targets[i] if targets is not None else None
#             image, target = self.resize(image, target)
#             images[i] = image
#             if targets is not None and target is not None:
#                 targets[i] = target

#         image_sizes = [image.shape[-2:] for image in images]
#         images = self.padding(images)

#         return image, targets

#     def resize(self, image: Tensor, target: Optional[Dict[str, Tensor]]):
#         h, w = image.shape[-2:]
#         image, target = self._resize_image_and_masks(image, size, float(self.max_size), target)

#         if target is None:
#             return image, target

        

#     def _resize_image_and_masks(
#         image: Tensor,
#         self_min_size: float,
#         self_max_size: float,
#         target: Optional[Dict[str, Tensor]]
#     ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
#         """
#         Args:
#         """
#         im_shape = torch.tensor(image.shape[-2:])
#         min_size = float(torch.min(im_shape))
#         max_size = float(torch.max(im_shape))
#         scale_factor = self_min_size / min_size
#         if max_size * scale_factor > self_max_size:
#             scale_factor = self_max_size / max_size

#         image = F.interpolate(
#             image[None],
#             scale_factor=scale_factor,
#             mode='bilinear',
#             recompute_scale_factor=True,
#             align_corners=False
#         )[0]

#         if target is None:
#             return image, target

#         if 'masks' in target:
#             mask = target['masks']
#             mask = F.interpolate(
#                 mask[:, None].float(),
#                 scale_factor=scale_factor,
#                 recompute_scale_factor=True
#             )[:, 0].byte()
#             target['masks'] = mask

#         return image, target

#     def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
#         """
#         Args:

#         """
#         ratios = [
#             torch.tensor(s, dtype=torch.float32, device=boxes.device) /
#             torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
#             for s, s_orig in zip(new_size, original_size)
#         ]

#         ratio_height, ratio_width = ratios
#         xmin, ymin, xmax, ymax = boxes.unbind(1)

#         xmin = xmin * ratio_width
#         xmax = xmax * ratio_width
#         ymin = ymin * ratio_height
#         ymax = ymax * ratio_height

#         return torch.stack((xmin, ymin, xmax, ymax), dim=1)

#     def max_by_axis(self, tensor_shapes):
#         """
#         tensor_shape [[3, 1920, 1080], [3, 1270, 720]]
#         return [3, 1920, 1080]
#         # TODO: 이렇게 forloop써서 해야하는가????
#         """
#         maxes = tensor_shapes[0]
#         for shape in tensor_shapes[1:]:
#             for index, item in enumerate(shape):
#                 maxes[index] = max(maxes[index], item)

#         return maxes

#     def padding(self, images: List[Tensor], size_divisible: int = 32):
#         # TODO: size_divisible은 왜 써야하는가? 설정한 min, max_size가 800, 1333인데 
#         # 1344 800으로 나가야하는 이유가 있는가?? 어차피 다시 resize밖에서 시키는데???
#         max_size = self.max_by_axis([list(img.shape) for img in images])
#         stride = float(size_divisible)
#         max_size = list(max_size)
#         max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
#         max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
#         print(max_size)

#         batch_shape = [len(images)] + max_size
#         batched_imgs = images[0].new_full(batch_shape, 0)
#         # TODO: padding 좌측상단부터 말고 센터를 기준으로 
#         for img, pad_img in zip(images, batched_imgs):
#             pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

#         return batched_imgs


class Compose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        image: ndarray,
        targets: Dict[str, ndarray]
    ) -> Tuple[ndarray, Dict[str, ndarray]]:
        for t in self.transforms:
            image, targets = t(image, targets)

        return image, targets


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.channel_map = {c: idx for idx, c in enumerate('BGR')}
        self.channel_permutation = [self.channel_map[c] for c in 'RGB']

    def __call__(self, image, targets):
        image = transforms.Normalize(
            self.mean, self.std)(image)
        # image = image[self.channel_permutation, :, :]
        return image, targets


class ToTensor:
    def __init__(self) -> None:
        self.dtype = {
            'boxes': torch.float32,
            'masks': torch.uint8,
            'scores': torch.int64,
        }

    def __call__(
        self,
        image: ndarray,
        targets: Dict[str, ndarray]
    ) -> Tuple[ndarray, Dict[str, Tensor]]:
        image = torch.as_tensor(image.transpose((2, 0, 1)), dtype=torch.float32)
        for key, value in targets.items():
            targets[key] = torch.as_tensor(value)

        return image, targets


class RandomFlip:
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        h, _, _ = image.shape
        if np.random.random() < self.p:
            image = image[::-1, :]
            targets['masks'] = targets['masks'][:, ::-1, :]
            boxes = targets['boxes'].copy()
            targets['boxes'][:, 1::2] = h - boxes[:, 3::-2]

        return image, targets


class RandomRotation:
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if np.random.random() < self.p:
            old_height, old_width, _ = image.shape
            k = np.random.randint(4)
            image = np.rot90(image, k)

            if targets['boxes'] is not None:
                boxes = targets['boxes'].copy()
                for _ in range(k):
                    boxes = np.array([[
                        box[1], old_width - 1 - box[2],
                        box[3], old_width - 1 - box[0]] for box in boxes])

                    old_width, old_height = old_height, old_width

            if targets['masks'] is not None:
                targets['masks'] = np.array(
                    [np.rot90(mask, k) for mask in targets['masks']])

        return image, targets


class Pad:
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.
    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, width, height, mean=None, p: float = 0.2):
        self.mean = mean
        self.width = width
        self.height = height

    def __call__(
        self,
        image: np.ndarray,
        targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        im_h, im_w, depth = image.shape

        expand_image = np.zeros(
            (self.height, self.width, depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image
        if targets['boxes'] is not None:
            expand_boxes = np.zeros_like(targets['boxes'])
            targets['boxes'] = expand_boxes

        if targets['masks'] is not None:
            expand_masks = np.zeros(
                (targets['masks'].shape[0], self.height, self.width),
                dtype=targets['masks'].dtype)
            expand_masks[:, :im_h, :im_w] = targets['masks']
            targets['masks'] = expand_masks

        return expand_image, targets


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5, p=0.2):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(
        self,
        image: np.ndarray,
        targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, targets


class RandomHue:
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(
        self,
        image: np.ndarray,
        targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        return image, targets


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5, p=0.2):
        self.lower = lower
        self.upper = upper
        self.p = p
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(
        self,
        image: np.ndarray,
        targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha

        return image, targets


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(
        self,
        image: np.ndarray,
        targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta

        return image, targets


# class ToTensor:
#     def __call__(
#         self,
#         image: np.ndarray,
#         targets: Dict[str, np.ndarray]
#     ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
#         return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), targets
