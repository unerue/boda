import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary


class Darknet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = None
        self.layers = self._make_layers(config)
        self.classifier = None
        self.linear = nn.Linear(1024*7*7, 4096)

    def forward(self, x):
        out = self.layers(x)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        for x in config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif len(x) == 4:
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=x[0], 
                        kernel_size=x[1], 
                        stride=x[2], 
                        padding=x[3]),
                    nn.BatchNorm2d(x[0]), 
                    nn.LeakyReLU(0.1)]
                in_channels = x[0]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=x[0], 
                        kernel_size=x[1], 
                        stride=1, 
                        padding=x[2]),
                    nn.BatchNorm2d(x[0]), 
                    nn.LeakyReLU(0.1)]
                in_channels = x[0]

        return nn.Sequential(*layers)


if __name__ == '__main__':
    config = [
        (64, 7, 2, 1), 'M', 
        (192, 3, 1, 1), 'M', 
        (128, 1, 1, 1), 
        (256, 3, 1, 1), 
        (256, 1, 1, 1), 
        (512, 3, 1, 1), 'M', 
        (256, 1, 1, 1), 
        (512, 3, 1, 1), 
        (256, 1, 1, 1), 
        (512, 3, 1, 1), 
        (256, 1, 1, 1), 
        (512, 3, 1, 1), 
        (256, 1, 1, 1), 
        (512, 3, 1, 1), 
        (512, 1, 1, 1), 
        (1024, 3, 1, 1), 'M',
        (512, 1, 1, 1), 
        (1024, 3, 1, 1), 
        (512, 1, 1, 1), 
        (1024, 3, 1, 1), 
        (1024, 3, 1, 1), 
        (1024, 3, 2, 1),
        (1024, 3, 1, 1), 
        (1024, 3, 1, 1), 
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Darknet(config).to(device)
    print(summary(net, input_data=(3, 448, 448), verbose=0))