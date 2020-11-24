# config.py
from ..arangement_base import PASCAL_CLASSES, COCO_CLASSES
from ..arangement_base import dataset_base, backbone_base, model_base
import os.path


pascal_voc_datset = dataset_base.copy({
    'name': 'pascal voc 2012',
    'train_images': './data/pascal_voc/VOC2012/train/JPEGImages',
    'test_images': './data/pasca_voc/VOC2012/test/JPEGImages',
    'train_labels': './data/pascal_voc/VOC2012/train_labels.txt',
    'test_labels': './data/pascal_voc/VOC2012/test_labels.txt',
    'class_names': PASCAL_CLASSES,
    'num_classes': 20
})

vgg16_backbone = backbone_base.copy({
    'name': 'vgg16',
    'pretrained': False, 
    'path': 'path/to/pretrained/weights',
    'backbone_layers': [
        [64, 64],
        ['M', 128, 128],
        ['M', 256, 256, 256],
        [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
        ['M', 512, 512, 512]]
})

ssd300_head = head_base.copy({
    
})

ssd_voc_vgg = model_base.copy({
    'name': 'yolov1 base',
    'dataset': pascal_voc_datset,
    'backbone': vgg16_backbone,
    'augmentation': None,
    'max_size': (448, 448),  # width, height
    'batch_size': 32,  # train batch size 64
    'max_iter': 120000,  # max_batches
    'lr': 0.0005,
    'momentum': 0.9,
    'decay': 0.0005,
    'lr_steps': (80000, 100000, 120000),
    'lr_scales': (2.5, 2.0, 2.0, 0.1, 0.1),
    # SSD
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
})

ssd_base_config = ssd_voc_vgg.copy()


# gets home dir cross platform
# HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS

# coco = {
#     'num_classes': 201,
#     'lr_steps': (280000, 360000, 400000),
#     'max_iter': 400000,
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [21, 45, 99, 153, 207, 261],
#     'max_sizes': [45, 99, 153, 207, 261, 315],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'COCO',
# }