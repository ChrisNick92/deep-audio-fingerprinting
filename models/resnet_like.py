import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention_mask import SpectroTemporalMask
from models.neural_fingerprinter import DivEncLayer

import torch.nn as nn
import torch.nn.functional as F

d=128


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, H, W, reduce=False, attention_flag=False):
        super(ResBlock, self).__init__()

        self.attention_flag = attention_flag
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if reduce:
            self.conv_2 = nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1
            )
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        else:
            self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.block = nn.Sequential(self.conv_1, self.batch_norm, self.relu, self.conv_2, self.batch_norm)
        if attention_flag and reduce:
            self.attention = SpectroTemporalMask(channels=out_channels, H=int(H // 2), W=int(W // 2))
        elif attention_flag and not reduce:
            self.attention = SpectroTemporalMask(channels=out_channels, H=H, W=W)

    def forward(self, x):

        out = self.block(x)
        if self.attention_flag:
            out = self.attention(out)
        x = self.conv(x)

        return self.relu(x + out)


class ResNet(nn.Module):

    def __init__(self, C, H, W):
        super(ResNet, self).__init__()

        self.first_conv = nn.Conv2d(in_channels=C, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.resblocks = nn.ModuleList()
        self.resblocks.append(ResBlock(in_channels=32, out_channels=32, attention_flag=True, H=H, W=W))
        C = 32
        for i in range(5):
            self.resblocks.append(
                ResBlock(in_channels=C, out_channels=2 * C, H=H, W=W, reduce=True, attention_flag=True)
            )
            C *= 2
            H /= 2
            W /= 2
        self.activation = nn.ELU()
        self.conv1d_1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.div_enc_layer = DivEncLayer(v=64)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.resblocks:
            x = block(x)
        
        x = self.div_enc_layer(x)
        return F.normalize(x, p=2, dim=1)
