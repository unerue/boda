import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo


class InterpolateModule(nn.Module):
	def __init__(self, *args, **kwdargs):
		super().__init__()
		self.args = args
		self.kwdargs = kwdargs

	def forward(self, inputs):
		return F.interpolate(inputs, *self.args, **self.kwdargs)


class Swish(nn.Module):
    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)


