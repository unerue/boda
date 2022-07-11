import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    """Swish https://arxiv.org/pdf/1905.02244.pdf"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """export-friendly version of nn.Hardswish()

    Return:
        x * F.hardsigmoid(x) for torchscript and CoreML
    """

    @staticmethod
    def forward(x):
        # for torchscript, CoreML and ONNX
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class MemoryEfficientSwish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


class Mish(nn.Module):
    """# Mish https://github.com/digantamisra98/Mish"""

    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


class FReLU(nn.Module):
    """FReLU https://arxiv.org/abs/2007.11824"""

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))
