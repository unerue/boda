import os
import json
from typing import Tuple, List, Dict, Any


PASCAL_CLASSES: Tuple[str] = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

COLORS: Tuple[Tuple[int]] = (
    (244, 67, 54),
    (233, 30, 99),
    (156, 39, 176),
    (103, 58, 183),
    (63, 81, 181),
    (33, 150, 243),
    (3, 169, 244),
    (0, 188, 212),
    (0, 150, 136),
    (76, 175, 80),
    (139, 195, 74),
    (205, 220, 57),
    (255, 235, 59),
    (255, 193, 7),
    (255, 152, 0),
    (255, 87, 34),
    (121, 85, 72),
    (158, 158, 158),
    (96, 125, 139))

COCO_CLASSES: Tuple[str] = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# COCO_LABEL_MAP: Dict[int, int] = { 
#     1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 
#     6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 
#     11: 11, 
    
#     13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 

#     21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
#     27: 25, 28: 26, 

#     31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 
# 
#     40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 

#     46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 
#     51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 
#     61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 

#     70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 
#     81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
# }



class Config:
    """Base Configuration
    """
    def __init__(self, config_dict: Dict[str, Any]):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict: Dict = {}):
        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict: Dict):
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, '=', v)


class PretrainedConfig:
    def __init__(self, *args, **kwargs):
        # backbone
        self.backbone = kwargs.get('backbone', None)

        # neck
        self.selected_layers = kwargs.get('selected_layers', list(range(1, 4)))
        self.pred_aspect_ratios = kwargs.get('pred_aspect_ratios', [[[1/2, 1, 2]]]*5)
        self.pred_scales = kwargs.get('pred_scales', [[24], [48], [96], [192], [384]])
        self.num_features

        # head
        
        # train or dataset?
        self.num_classes = kwargs.get('num_classes', 80)
        self.max_size = kwargs.get('max_size', 550)

        for k, v in kwargs:
            try:
                setattr()
            except AttributeError as e:
                print(k, v)

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: str, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            config_file = hf_bucket_url(
                pretrained_model_name_or_path, filename=CONFIG_NAME, revision=revision, mirror=None
            )

        try:
            # Load from URL or cache if already cached
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)

        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
            )
            raise EnvironmentError(msg)

        except json.JSONDecodeError:
            msg = (
                "Couldn't reach server at '{}' to download configuration file or "
                "configuration file is not a valid JSON file. "
                "Please check network or file content here: {}.".format(config_file, resolved_config_file)
            )
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(config_file, resolved_config_file))

        return config_dict, kwargs