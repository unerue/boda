import os
import copy
import json
from typing import Tuple, List, Dict, Any, Union


class BaseConfig:
    """
    Class attributes:
        model_type (str):
    Arguments:
        name_or_path (str):
    """
    config_name: str = ''

    def __init__(self, **kwargs):
        self.use_amp = kwargs.pop('use_amp', False)
        self.use_jit = kwargs.pop('use_jit', False)
        self.device = kwargs.pop('device', 'cpu')
        self.freeze_bn = kwargs.pop('freeze_bn', False)

        self.structures = kwargs.pop('structures', [])
        self.num_classes = kwargs.pop('num_classes', 80)
        self.min_size = kwargs.pop('min_size', None)
        self.max_size = kwargs.pop('max_size', None)
        if not isinstance(self.max_size, tuple):
            self.max_size = (self.max_size, self.max_size)

        self.num_grids = 0
        self.top_k = kwargs.pop('top_k', 5)
        self.score_thresh = kwargs.pop('score_thresh', 0.15)

        # backbone
        self.backbone_name = kwargs.pop('backbone_name', 'resnet101')
        self.backbone_structure = kwargs.pop('backbone_structure', None)

        # neck
        self.neck_name = kwargs.pop('neck_name', 'fpn')
        self.selected_layers = kwargs.pop('selected_layers', [1, 2, 3])
        self.aspect_ratios = kwargs.pop('aspect_ratios', [[[1/2, 1, 2]]]*5)
        self.scales = kwargs.pop('scales', [[24], [48], [96], [192], [384]])
        self.fpn_out_channels = kwargs.pop('fpn_out_channels', 256)

        # head
        self.anchors = kwargs.pop('anchors', None)

        for k, v in kwargs.items():
            try:
                setattr(k, v)
            except AttributeError as e:
                print(k, v, e)

    def __repr__(self):
        return f'{self.__class__.__name__} {self.to_dict()}'

    def to_json(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, 'model_name'):
            output['model_name'] = self.__class__.model_name
        return output

    def save_json(self, path: str):
        if os.path.isfile(path):
            raise AssertionError

        os.makedirs(path, exist_ok=True)
        # config_file = os.path.join(path, CONFIG_NAME)
        with open(path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json())

    def update(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: str, **kwargs):
        config, model_kwargs = cls.get_config_dict('', test='test')
        # raise NotImplementedError)
        print('Load config!')
        return config, model_kwargs
    
    @classmethod
    def _dict_from_json_file(self, model_name_or_path):
        return model_name_or_path
    
    @classmethod
    def get_config_dict(cls, model_name_or_path, **kwargs):
        config_dict = cls._dict_from_json_file(model_name_or_path)
        return config_dict, kwargs

    # @classmethod
    # def from_json(cls, json_file: str):
    #     with open(path, 'r') as json_file:
    #         config_dict = json.loads(json_file)
    #     config_dict = cls.dict_from_json_fiel(json_file)
    #     return cls(**config_dict)


    # @classmethod
    # def from_pretrained(cls, pretrained_model_or_path: str, **kwargs):
    #     raise NotImplementedError

    # @classmethod
    # def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    #     cache_dir = kwargs.pop('cache_dir', None)
    #     force_download = kwargs.pop('force_download', False)
    #     resume_download = kwargs.pop('resume_download', False)
    #     proxies = kwargs.pop("proxies", None)
    #     local_files_only = kwargs.pop("local_files_only", False)
    #     revision = kwargs.pop("revision", None)

    #     if os.path.isdir(pretrained_model_name_or_path):
    #         config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
    #     elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
    #         config_file = pretrained_model_name_or_path
    #     else:
    #         config_file = hf_bucket_url(
    #             pretrained_model_name_or_path, filename=CONFIG_NAME, revision=revision, mirror=None
    #         )

    #     try:
    #         # Load from URL or cache if already cached
    #         resolved_config_file = cached_path(
    #             config_file,
    #             cache_dir=cache_dir,
    #             force_download=force_download,
    #             proxies=proxies,
    #             resume_download=resume_download,
    #             local_files_only=local_files_only,
    #         )
    #         # Load config dict
    #         config_dict = cls._dict_from_json_file(resolved_config_file)

    #     except EnvironmentError as err:
    #         logger.error(err)
    #         msg = (
    #             f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
    #             f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
    #             f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
    #         )
    #         raise EnvironmentError(msg)

    #     except json.JSONDecodeError:
    #         msg = (
    #             "Couldn't reach server at '{}' to download configuration file or "
    #             "configuration file is not a valid JSON file. "
    #             "Please check network or file content here: {}.".format(config_file, resolved_config_file)
    #         )
    #         raise EnvironmentError(msg)

    #     if resolved_config_file == config_file:
    #         logger.info("loading configuration file {}".format(config_file))
    #     else:
    #         logger.info("loading configuration file {} from cache at {}".format(config_file, resolved_config_file))

    #     return config_dict, kwargs