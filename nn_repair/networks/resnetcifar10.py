import torch
from torch import nn


def _conv3x3(in_channels: int, double_channels: bool = False):
    if double_channels:
        out_channels = 2 * in_channels
        stride = 2
    else:
        out_channels = in_channels
        stride = 1

    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    return layer


def _conv1x1(in_channels: int, out_channels: int):
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=2,
        bias=False
    )
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    return layer


def _bn(channels: int):
    return nn.BatchNorm2d(channels)


def _downsample(in_channels, out_channels):
    return nn.Sequential(
        _conv1x1(in_channels, out_channels),
        _bn(out_channels)
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(BasicBlock, self).__init__()
        if in_channels == out_channels:
            self.conv1 = _conv3x3(in_channels)
            self.downsample = None
        else:
            self.conv1 = _conv3x3(in_channels, double_channels=True)
            self.downsample = _downsample(in_channels, out_channels)
        self.bn1 = _bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels)
        self.bn2 = _bn(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y = y + x
        return self.relu(y)


class ResNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, size: int):
        super(ResNetLayer, self).__init__()
        self.blocks = nn.ModuleList(
            [BasicBlock(in_channels, out_channels)]
            + [BasicBlock(out_channels, out_channels) for _ in range(size - 1)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ResNetCIFAR10(nn.Module):
    """
    An implementation of ResNets for CIFAR-10 [HeEtAl2015]_.
    For ImageNet ResNets, use :func:`torchvision.models.resnet50` and
    the other classes from :code:`torchvision.models`.
    The implementation is based on these models, as well as [HeEtAl2015]_.

    .. [HeEtAl2015] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun:
       Deep Residual Learning for Image Recognition. CoRR abs/1512.03385 (2015)
    """
    def __init__(self, size: int = 3):
        """
        Create a CIFAR-10 ResNet.

        :param size: The value :math:`n` in [HeEtAl2015]_, section 4.2.
        """
        super(ResNetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = _bn(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = ResNetLayer(16, 16, size)
        self.layer2 = ResNetLayer(16, 32, size)
        self.layer3 = ResNetLayer(32, 64, size)
        # Using ordinary AvgPool allows using ERAN
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_features=64, out_features=10)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.constant_(self.fc.bias, val=0.0)

    def forward(self, x):
        x = self.conv1(x)  # 16 x 32 x 32
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 16 x 32 x 32

        x = self.layer1(x)  # 16 x 32 x 32
        x = self.layer2(x)  # 32 x 16 x 16
        x = self.layer3(x)  # 64 x 8 x 8

        x = self.avgpool(x)  # 64 x 1 x 1
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
