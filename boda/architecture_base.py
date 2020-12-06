import torch
from torch import nn, Tensor


class BaseModel(nn.Module):
    """Base Model for 
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # require_pytorch(self)
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        pass
        # require_pytorch(self)

    @classmethod
    def check_inputs(cls, inputs):
        """
        Argument:
            inputs (List[Tensor]): Size([C, H, W])
        Return:
            outputs (Tensor): Size([B, C, H, W])
        """
        for image in inputs:
            if isinstance(image, Tensor):
                if image.dim() != 3:
                    raise ValueError(f'images is expected to be 3d tensors of shape [C, H, W] {image.shape}')
            else:
                raise ValueError('Expected image to be Tensor.')

        if isinstance(inputs, list):
            inputs = torch.stack(inputs)

        return inputs
