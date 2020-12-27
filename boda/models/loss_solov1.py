import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..architecture_base import LossFunction
from ..utils.bbox import jaccard, cxcywh_to_xyxy
from ..utils.loss import log_sum_exp


class Matcher:
    def __init__(self, config, threshold, variances):
        self.config = config

    def __call__(
        self,
        pred_boxes,
        pred_priors,
        true_boxes,
        idx
    ) -> Tensor:
        raise NotImplementedError

    def encode(self, matched, priors, variances):
        raise NotImplementedError

    def decode(self, pred_boxes, pred_priors, variances):
        raise NotImplementedError


class Solov1Loss(LossFunction):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config

    def forward(self, inputs, targets):
        self.check_targets(targets)
        targets = self.copy_targets(targets)

        pred_boxes = inputs['boxes']
        pred_scores = inputs['scores']
        pred_priors = inputs['priors']

        true_boxes = targets['boxes']
        true_labels = targets['labels']

        return
        