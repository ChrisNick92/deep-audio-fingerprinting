import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.attention_mask import SpectroTemporalMask

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

d = 128
h = 1024
u = 32
v = int(h / d)
channel_sequence_1 = [d, 2 * d, 2 * d]
channel_sequence_2 = [4 * d, 4 * d, h]


class SC(nn.Module):

    def __init__(
        self,
        input_shape,
        in_channels,
        out_channels,
        kernel_sizes,
        padding_sizes,
        stride_sizes,
        norm="layer_norm2d",
        attention=False
    ):
        super(SC, self).__init__()

        C, H, W = input_shape
        out_H = int(np.floor((H + 2 * padding_sizes[1][0] - kernel_sizes[1][0]) / stride_sizes[1][0] + 1))
        out_W = int(np.floor((W + 2 * padding_sizes[0][1] - kernel_sizes[0][1]) / stride_sizes[0][1] + 1))

        self.separable_conv2d_1x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[0],
            stride=stride_sizes[0],
            padding=padding_sizes[0]
        )
        self.separable_conv2d_3x1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[1],
            stride=stride_sizes[1],
            padding=padding_sizes[1]
        )

        if norm == 'layer_norm1d':
            self.BN_1x3 = nn.LayerNorm(out_W)
            self.BN_3x1 = nn.LayerNorm(out_W)
        elif norm == 'layer_norm2d':
            self.BN_1x3 = nn.LayerNorm((out_channels, H, out_W))
            self.BN_3x1 = nn.LayerNorm((out_channels, out_H, out_W))
        elif norm == 'batch_norm':
            self.BN_1x3 = nn.BatchNorm2d(out_channels)
            self.BN_3x1 = nn.BatchNorm2d(out_channels)

        if attention:
            self.mask = SpectroTemporalMask(channels=out_channels, H=out_H, W=out_W)
            self.separable_conv2d = nn.Sequential(
                self.separable_conv2d_1x3, nn.ReLU(), self.BN_1x3, self.separable_conv2d_3x1, nn.ReLU(), self.BN_3x1,
                self.mask
            )
        else:
            self.separable_conv2d = nn.Sequential(
                self.separable_conv2d_1x3, nn.ReLU(), self.BN_1x3, self.separable_conv2d_3x1, nn.ReLU(), self.BN_3x1
            )

    def forward(self, x):
        return self.separable_conv2d(x)


class Encoder(nn.Module):

    def __init__(
        self,
        shapes=[
            (1, 256, 32), (d, 128, 16), (d, 64, 8), (2 * d, 32, 4), (2 * d, 16, 2), (4 * d, 8, 1), (4 * d, 4, 1),
            (h, 2, 1)
        ],
        channel_seq=[1, d, d, 2 * d, 2 * d, 4 * d, 4 * d, h, h],
        kernel_seq=[[(1, 3), (3, 1)] for i in range(8)],
        stride_seq=[[(1, 2), (2, 1)] for i in range(8)],
        pad_seq=[[(0, 1), (1, 0)] for i in range(8)],
        attention=False
    ):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential()
        for i in range(len(channel_seq) - 1):
            self.encoder.append(
                SC(
                    input_shape=shapes[i],
                    in_channels=channel_seq[i],
                    out_channels=channel_seq[i + 1],
                    kernel_sizes=kernel_seq[i],
                    padding_sizes=pad_seq[i],
                    stride_sizes=stride_seq[i],
                    attention=attention
                )
            )

    def forward(self, x):
        x = self.encoder(x)
        return torch.flatten(x, start_dim=1)


class DivEncLayer(nn.Module):

    def __init__(self, q=d, unit_dim=[32, 1]):
        super(DivEncLayer, self).__init__()
        self.split_fc_layers = nn.ModuleList()
        self.q = q
        self.unit_dim = unit_dim
        self._construct_layers()

    def _construct_layers(self):
        for i in range(self.q):
            seq = nn.Sequential()
            seq.append(nn.Linear(v, self.unit_dim[0]))
            seq.append(nn.ELU())
            seq.append(nn.LayerNorm(self.unit_dim[0]))
            seq.append(nn.Linear(self.unit_dim[0], self.unit_dim[1]))
            self.split_fc_layers.append(seq)

    def _split_encoding(self, x_slices):
        out = []
        for i in range(self.q):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        return torch.concat(out, dim=1)

    def forward(self, x):
        # x: BxD, D=1024
        x = torch.reshape(x, (x.shape[0], self.q, -1))
        return self._split_encoding(x)


class Neural_Fingerprinter(nn.Module):

    def __init__(self, attention=False) -> None:
        super(Neural_Fingerprinter, self).__init__()
        self.encoder = Encoder(attention=attention)
        self.div_enc_layer = DivEncLayer()

    def forward(self, x):
        x = self.encoder(x)
        x = self.div_enc_layer(x)
        return F.normalize(x, p=2, dim=1)