import torch.nn as nn
import torch.nn.functional as F
import torch


class SpectroTemporalMask(nn.Module):

    def __init__(self, channels, H, W, S=100):
        super(SpectroTemporalMask, self).__init__()
        self.channels = channels
        # Height -> Frequencies, Width -> Time stamps
        # C x F x T
        self.F = H
        self.T = W
        self.S = S

        self.W_temp = nn.Parameter(torch.ones(self.channels, self.T, 1, dtype=torch.float32), requires_grad=True)
        self.W_spec = nn.Parameter(torch.ones(self.channels, 1, self.F, dtype=torch.float32), requires_grad=True)

    def forward(self, x):

        X_temp = torch.einsum("...cft,ctf->...cf", x, self.W_temp)
        A_temp = F.softmax(X_temp, dim=2)

        X_spec = torch.einsum("...cft,ctf->...ct", x, self.W_spec)
        A_spec = F.softmax(X_spec, dim=2)

        A_mask = torch.einsum("...cf,...ct->cft", A_temp, A_spec)

        return A_mask * x * self.S


class ParallelAttentionMask(nn.Module):

    def __init__(self, C, H, W):
        super(ParallelAttentionMask, self).__init__()

        self.C, self.H, self.W = C, H, W
        self.weights = nn.Parameter(data=torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.conv_T = nn.Conv2d(in_channels=C, out_channels=1, kernel_size=1)
        self.conv_F = nn.Conv2d(in_channels=C, out_channels=1, kernel_size=1)
        self.avg_T = nn.AvgPool2d(kernel_size=(H, 1))
        self.avg_F = nn.AvgPool2d(kernel_size=(1, W))

    def forward(self, x):
        V_T = self.conv_T(x)
        V_F = self.conv_F(x)

        v_T = torch.sigmoid(self.avg_T(V_T))
        v_F = torch.sigmoid(self.avg_F(V_F))

        U_T = x * v_T
        U_F = x * v_F

        a, b, c = self.weights

        return a * U_T + b * U_F + c * x