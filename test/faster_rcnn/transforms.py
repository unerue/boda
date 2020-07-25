from torchvision import transforms
import torch
from skimage import transform as sktransform
import random

def image_normalize(img):
    """
    Normalize an image to match the input distribution of pytorch pretrained model.
    Note: the pixel value of the image should be in range [0,1], if the original image
          is in range [0,255], do not forget to div it by 255 before pass it to this function.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def adjust_image_size(image, min_size=600, max_size=1000):
    C, H, W = image.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    image = sktransform.resize(image, (C, H * scale, W * scale), mode='reflect')
    return image


def resize_bbox(bbox, image_size_in, image_size_out):
    """
    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    """
    bbox = bbox.copy()
    y_scale = float(image_size_out[0]) / image_size_in[0]
    x_scale = float(image_size_out[1]) / image_size_in[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def random_flip(image, bbox, vertical_random=False, horizontal_random=False):
    vertical_flip, horizontal_flip = False, False
    H,W = image.shape[1], image.shape[2]
    bbox = bbox.copy()

    if vertical_random:
        vertical_flip =  random.choice([True, False])
    if horizontal_random:
        horizontal_flip = random.choice([True, False])

    if vertical_flip:
        image = image[:,::-1,:]
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max

    if horizontal_flip:
        image = image[:,:,::-1]
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    
    return image, bbox
