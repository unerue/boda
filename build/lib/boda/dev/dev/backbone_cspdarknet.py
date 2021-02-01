from torch import nn, Tensor


class ResidualBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, num_blocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(num_blocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBnActivation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(ConvBnActivation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x  # TODO: ?!!!!
            for res in module:
                h = res(h)

            if self.shortcut:
                x = x + h
            else:
                x = h
            # x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(3, 32, 3, 1, 'mish')

        self.conv2 = ConvBnActivation(32, 64, 3, 2, 'mish')
        self.conv3 = ConvBnActivation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = ConvBnActivation(64, 64, 1, 1, 'mish')

        self.conv5 = ConvBnActivation(64, 32, 1, 1, 'mish')
        self.conv6 = ConvBnActivation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = ConvBnActivation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = ConvBnActivation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(64, 128, 3, 2, 'mish')
        self.conv2 = ConvBnActivation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = ConvBnActivation(128, 64, 1, 1, 'mish')

        self.resblock = ResidualBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = ConvBnActivation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = ConvBnActivation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(128, 256, 3, 2, 'mish')
        self.conv2 = ConvBnActivation(256, 128, 1, 1, 'mish')
        self.conv3 = ConvBnActivation(256, 128, 1, 1, 'mish')

        self.resblock = ResidualBlock(ch=128, nblocks=8)
        self.conv4 = ConvBnActivation(128, 128, 1, 1, 'mish')
        self.conv5 = ConvBnActivation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(256, 512, 3, 2, 'mish')
        self.conv2 = ConvBnActivation(512, 256, 1, 1, 'mish')
        self.conv3 = ConvBnActivation(512, 256, 1, 1, 'mish')

        self.resblock = ResidualBlock(ch=256, nblocks=8)
        self.conv4 = ConvBnActivation(256, 256, 1, 1, 'mish')
        self.conv5 = ConvBnActivation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5