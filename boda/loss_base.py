from typing import List, Dict
from torch import nn, Tensor


class LoseFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def copy_targets(cls, targets):
        if targets is not None:
            targets_copy: List[Dict[str, Tensor]] = []
            for target in targets:
                _target: Dict[str, Tensor] = {}
                for key, value in target.items():
                    _target[key] = value
                targets_copy.append(_target)
            targets = targets_copy

        return targets
        
    @classmethod
    def check_targets(cls, targets: List[Dict[str, Tensor]]):
        for target in targets:
            if isinstance(target['boxes'], Tensor):
                if target['boxes'].dim() != 2 or target['boxes'].size(1) != 4:
                    raise ValueError('Expected target boxes to be a tensor of [N, 4].')
                elif target['labels'].dim () != 1:
                    raise ValueError('Expected target boxes to be a tensor of [N].')
            else:
                raise ValueError('Expected target boxes to be Tensor.')
            break