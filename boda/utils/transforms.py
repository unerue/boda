import random
from typing import Tuple, List, Dict
import numpy as np
from torch import Tensor


class Compose:
    def __init__(self, transforms: List[object]):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for transform in self.transforms:
            image, boxes, labels = transform(image, boxes, labels)
        return image, boxes, labels


class ToTensor:
    def __call__(self, image, boxes=None, labels=None):
        image = torch.from_numpy(image.transpose(2, 0, 1))
        boxes = torch.from_numpy(boxes)
        return image, boxes, labels


class Resize:
    def __init__(self, size: Tuple[int, int]):
        if not isinstance(size, tuple):
            raise ValueError

        self.size = size

    def __call__(self, image, boxes: List[List[float]], labels=None):
        w, h = image.size

        image = image.resize(self.size)
        boxes[:, [0, 2]] *= self.size[0] / w
        boxes[:, [1, 3]] *= self.size[1] / h

        return image, boxes, labels


class Normalize:
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.
    """
    def __init__(self, mean: List[float], std: List[float], normlize: bool = True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std,  dtype=np.float32)

    def __call__(self, image, boxes, labels=None):
        image = np.asarray(image, dtype=np.float32)
        image = (image - self.mean) / self.std    
        image = image / 255

        return image, boxes, labels 


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, p=0.5):
        assert upper >= lower, 'contrast upper must be >= lower.'
        assert lower >= 0, 'contrast lower must be non-negative.'

        self.lower = lower
        self.upper = upper
        self.p = p
        
    def __call__(self, image, boxes=None, labels=None):
        if random.random() > self.p:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0, p=0.5):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.p = p

    def __call__(self, image, boxes=None, labels=None):
        if random.random() > self.p:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, p=0.5):
        assert upper >= lower, 'contrast upper must be >= lower.'
        assert lower >= 0, 'contrast lower must be non-negative.'
        self.lower = lower
        self.upper = upper
        self.p = p
        
    def __call__(self, image, boxes=None, labels=None):
        if random.random() > self.p:
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha

        return image, boxes, labels


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        height , _ , _ = image.shape
        if random.random() > self.p:
            image = image[::-1, :]
            masks = masks[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]

        return image, boxes, labels



