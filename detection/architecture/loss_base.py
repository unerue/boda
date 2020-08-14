from typing import List, Dict
from torch import Tensor


def _check_targets(targets: List[Dict[str, Tensor]]):
    for target in targets:
        if isinstance(target['boxes'], Tensor):
            if target['boxes'].dim() != 2 or target['boxes'].size(1) != 4:
                raise ValueError('Expected target boxes to be a tensor of [N, 4].')
            elif target['labels'].dim () != 2:
                raise ValueError('Expected target boxes to be a tensor of [N, C].')
        else:
            raise ValueError('Expected target boxes to be Tensor.')


def _copy_targets(targets):
    if targets is not None:
        targets_copy: List[Dict[str, Tensor]] = []
        for t in targets:
            target: Dict[str, Tensor] = {}
            for k, v in t.items():
                target[k] = v
            targets_copy.append(target)
        targets = targets_copy