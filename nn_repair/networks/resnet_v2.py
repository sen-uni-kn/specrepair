from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetStage(nn.Module):
    """
    One group of ResNetv2 layers.
    """
    def __init__(self, size: int, in_channels: int, first: bool = False):
        super().__init__()
        assert size >= 1
        self.size = size

        if first:
            first_stride = 1
            first_padding = 0
            self.out_channels = in_channels * 4
        else:
            first_stride = 2
            first_padding = 16
            self.out_channels = in_channels * 2

        block0_layers = []
        if not first:
            block0_layers += [
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
            ]
        block0_layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=(1, 1),
                      stride=(first_stride, first_stride), padding=(first_padding, first_padding))
        )
        block0_layers += [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=(1, 1))
        ]
        self.block0 = nn.Sequential(*block0_layers)
        self.expand_match = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels,
                                      kernel_size=(1, 1),
                                      stride=(first_stride, first_stride), padding=(first_padding, first_padding))

        for i in range(size-1):
            block = nn.Sequential(
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.out_channels, out_channels=in_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=(1, 1))
            )
            self.add_module(f"block{i+1}", block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block0(x) + self.expand_match(x)
        for i in range(self.size-1):
            y = getattr(self, f'block{i+1}')(y) + y
        return y


class ResNetv2(nn.Module):
    """
    A Resnet v2 implementation.

    Sources:
     - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Deep Residual Learning for Image Recognition.
       CoRR abs/1512.03385 (2015)
     - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Identity Mappings in Deep Residual Networks.
       CoRR abs/1603.05027 (2016)
    Code sources:
     - https://github.com/Jianbo-Lab/HSJA/blob/master/resnet.py
     - https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
     - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, stage_size: int = 6, first_expansion: int = 16):
        super().__init__()
        assert stage_size > 0
        assert first_expansion > 0

        self.stage_size = stage_size
        self.input_shape = input_shape

        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=first_expansion, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(first_expansion),
            nn.ReLU()
        )
        self.stage0 = ResnetStage(size=stage_size, in_channels=first_expansion, first=True)
        self.stage1 = ResnetStage(size=stage_size, in_channels=self.stage0.out_channels, first=True)
        self.stage2 = ResnetStage(size=stage_size, in_channels=self.stage1.out_channels, first=True)

        self.final_batch_norm = nn.BatchNorm2d(self.stage2.out_channels)
        # all layers maintain the width and height of the input
        self.final_pool = nn.AvgPool2d(kernel_size=(input_shape[1], input_shape[2]))
        self.final_linear = nn.Linear(self.stage2.out_channels, num_classes)

        def _init_weights(module):
            if type(module) == nn.Linear or type(module) == nn.Conv2d:
                nn.init.kaiming_normal_(module.weight)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.final_batch_norm(x)
        x = F.relu(x)
        x = self.final_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.final_linear(x)
        return x

    @property
    def depth(self):
        return 9 * self.stage_size + 2
