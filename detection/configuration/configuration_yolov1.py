from typing import Tuple, List, Dict
from .configuration_base import Config
from .configuration_base import PASCAL_CLASSES, COCO_CLASSES, COCO_LABEL_MAP


dataset_base = Config({
    'name': 'base dataset',

    'train_images': './path/to/images',
    'valid_images': './path/to/images',
    
    'train_labels': './path/to/labels/',
    'valid_labels': './path/to/labels/',

    'class_names': Tuple[str],
    'num_classes': int,
})

backbone_base = Config({
    'name': 'base backbone',
    'pretrained': bool,
    'path': 'path/to/pretrained/weights',
})

augmentation_base = Config({
    'saturation': 1.5,
    'exposure': 1.5,
    'hue': 0.1,
})

model_base = Config({
    'name': 'base model',
    'path': 'path/to/pretrained/weights',
    'dataset': dataset_base,
    'backbone': backbone_base,
    'batch_size': int,
    'max_size': Tuple[int, int],
    'lr': float,
})

pascal_voc_datset = dataset_base.copy({
    'name': 'pascal voc 2007',
    'train_images': './data/pascal_voc/VOC2012/train/JPEGImages',
    'test_images': './data/pasca_voc/VOC2012/test/JPEGImages',
    'train_labels': './data/pascal_voc/VOC2012/train_labels.txt',
    'test_labels': './data/pascal_voc/VOC2012/test_labels.txt',
    'class_names': PASCAL_CLASSES,
    'num_classes': 20,
})

darknet9_backbone = backbone_base.copy({
    'name': 'darknet9',
    'pretrained': False, 
    'path': 'path/to/pretrained/weights',
})

yolov1_base_config = model_base.copy({
    'name': 'yolov1 base',
    'dataset': pascal_voc_datset,
    'backbone': darknet9_backbone,
    'augmentation': None,
    'max_size': (448, 448),  # width, height
    'batch_size': 1,  # train batch size 64
    'max_iter': 40000,  # max_batches
    'lr': 0.0005,
    'momentum': 0.9,
    'decay': 0.0005,
    'lr_steps': (200, 400, 600, 20000, 30000),
    'lr_scales': (2.5, 2.0, 2.0, 0.1, 0.1),
    'num_boxes': 2,
    'grid_size': 7,
    'object_scale': 1,
    'noobject_scale': 0.5, 
    'class_scale': 1,
    'coord_scale': 5,
    'jitter': 0.2, 
    'rescore': 1,
    'sqrt': 1,
    'num': 3,  # ????
    'lambda_coord': 5.0,
    'lambda_noobj': 0.5,
})

yolov1_config = yolov1_base_config.copy()

class Yolov1Config(Config):
    def __init__(self, config):
        global yolov1_config
        yolov1_config.replace(eval(config))



