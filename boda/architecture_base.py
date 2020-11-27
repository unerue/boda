from torch import nn, Tensor


PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

try:
    USE_TORCH = os.environ.get('USE_TORCH', 'AUTO').upper()
    if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES not in ENV_VARS_TRUE_VALUES:
        import torch

        is_torch_available = True
        logger.info(f'PyTorch version {torch.__version__}')
    else:
        is_torch_available = False
except:
    is_torch_available = False


def is_torch_available():
    return _torch_available


def require_pytorch(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_torch_available():
        raise ImportError(PYTORCH_IMPORT_ERROR.foramt(name))


class PreTrainedModel:
    def __init__(self, *args, **kwargs):
        require_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        require_pytorch(self)

    @classmethod
    def check_inputs(cls, inputs):
        for image in inputs:
            if isinstance(image, Tensor):
                if image.dim() != 3:
                    raise ValueError(f'images is expected to be 3d tensors of shape [C, H, W] {image.shape}')
            else:
                raise ValueError('Expected image to be Tensor.')
        if not isinstance(inputs, Tensor):
            inputs = torch.stack(inputs)

        return inputs
