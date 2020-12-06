import os
from typing import Tuple, List, Dict, Union

import torch
from torch import nn, Tensor


class BaseModel(nn.Module):
    """Base Model for 
    """
    checked_inputs = True
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @classmethod
    def check_inputs(cls, inputs):
        """
        Argument:
            inputs (List[Tensor]): Size([C, H, W])
        Return:
            outputs (Tensor): Size([B, C, H, W])
        """
        if cls.checked_inputs:
            print('Check!!')
            for image in inputs:
                if isinstance(image, Tensor):
                    if image.dim() != 3:
                        raise ValueError(f'images is expected to be 3d tensors of shape [C, H, W] {image.size()}')
                else:
                    raise ValueError('Expected image to be Tensor.')
            cls.checked_inputs = False

        if isinstance(inputs, list):
            inputs = torch.stack(inputs)

        return inputs

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike], *args, **kwargs):
        model_name_or_path = str(model_name_or_path)
        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, CONFIG_NAME)

    def _check_pretrained_model_is_valid(cls, model_name_or_path):
        # if model_name_or_path not in 
        pass



        
    #     pass

    # @classmethod
    # def get_config_dict(cls, model_name_or_path, **kwargs):

