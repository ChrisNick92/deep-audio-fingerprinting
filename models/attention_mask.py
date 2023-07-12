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

        self.W_temp = nn.Parameter(torch.randn((self.channels, self.T, 1)), requires_grad=True)
        self.W_spec = nn.Parameter(torch.randn((self.channels, 1, self.F)), requires_grad=True)

    def forward(self, x):

        X_temp = torch.einsum("...cft,ctf->...cf", x, self.W_temp)
        A_temp = F.softmax(X_temp, dim=2)
        
        X_spec = torch.einsum("...cft,ctf->...ct", x, self.W_spec)
        A_spec = F.softmax(X_spec, dim=2)
        
        A_mask = torch.einsum("...cf,...ct->cft", A_temp, A_spec)
        
        return A_mask * x * self.S