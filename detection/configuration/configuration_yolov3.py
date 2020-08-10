from .configuration_base import Config
from .configuration_base import PASCAL_CLASSES, COCO_CLASSES, COCO_LABEL_MAP
from typing import List, Tuple, Dict

import sys
import pprint


dataset_base = Config({
    'name': str,

    'train_images_dir': str,
    'valid_images_dir': str,

    'train_label_path': str,
    'valid_label_path': str,

    'train_label_dir': str,
    'valid_label_dir': str,

    'class_names': Tuple[str],
    'num_classes': int,
})

pascal_dataset = dataset_base.copy({
    'name': 'Pascal SBD 2012',
    'train_images_dir': './data/sbd/img/',
    'valid_images_dir': './data/sbd/img/',
    'train_label_path': './data/sbd/pascal_sbd_train.json',
    'valid_label_path': './data/sbd/pascal_sbd_val.json',
    'class_names': PASCAL_CLASSES,
    'num_classes': 20,
})

backbone_base = Config({
    'name': 'darknet50',
    'path': 'path/to/pretrained/weights/darknet50.weight',
})

augmentation_base = Config({})

yolov3_base = Config({
    'max_size': 416,
    'num_boxes': 3,

    'selected_layers': 3,

    'burn_in': 1000,
    'max_batches': 500200,
    'batch_size': 16,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'decay': 0.0005,
    'weight_decay': 0,
    'policy': 'step', # Scheduler
    'steps': (400000, 450000),
    'scales': (0.1, 0.1),

    'downsample': [[128, 256], [256, 512]],

    'num_anchors': 9,
    'anchors': [
        (10, 13), (16, 30), (33, 23), 
        (30, 61), (62, 45), (59, 119),
        (116, 90), (156, 198), (373, 326)
    ],
    'masks': [(0, 1, 2), (3, 4, 5), (6, 7, 8)],

    'jitter': (.3, .3, .3),
    'ignore_thresh': (.7, .7, .7),
    'truth_thresh': (1, 1, 1),
})


yolov3_base_darknet_pascal = yolov3_base.copy({
    'backbone': backbone_base,
    'dataset': pascal_dataset,
})

# pprint.pprint(yolov3_darknet_pascal.print())
# pprint.pprint(yolov3_darknet_pascal.dataset.print())
# pprint.pprint(yolov3_darknet_pascal.backbone.print())

# sys.exit(0)




"""
Pascal VOC or MS COCO 
[x_min, y_min, x_max, y_max]
[x, y, w, h]


"""

"""
image: a PIL Image of size (H, W)
target: a dict containing the following fields
boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, 
                            ranging from 0 to W and 0 to H
labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, 
                           and is used during evaluation
area (Tensor[N]): The area of the bounding box. 
                  This is used during evaluation with the COCO metric, 
                  to separate the metric scores between small, medium and large boxes.

iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
# (optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
# (optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, 
# it contains the K keypoints in [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible. Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, and you should probably adapt references/detection/transforms.py for your new keypoint representation
"""