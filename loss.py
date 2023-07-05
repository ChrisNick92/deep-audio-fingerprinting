import torch.nn as nn
import torch
import torch.nn.functional as F


class NTxent_Loss(nn.Module):
    """Custom implementation of NTxent Loss as in https://arxiv.org/abs/2010.11910"""

    def __init__(self, T: float = 0.05):
        """Input assumes to be a matrix of 2N x d. First N rows are clean sounds, last the augmented.
        
        Args:
            N (int): Number of clean samples per batch
            T (float): The temperature
        """
        super(NTxent_Loss, self).__init__()

        self.T = T
        self.cross_entropy = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, X):

        N = X.shape[0] // 2
        A_ij = (X @ X.T) / self.T
        zero_out = torch.diag(-torch.inf * torch.ones(2 * N)).to(self.device)
        A_ij += zero_out
        labels = torch.cat((torch.arange(N, 2 * N), torch.arange(N))).to(self.device)

        return self.cross_entropy(A_ij, labels)


class NTxent_Loss_2(nn.Module):

    def __init__(self, n_org: int, n_rep: int, T: float = 0.05, device: str = None):
        super(NTxent_Loss_2, self).__init__()
        self.T = T
        self.n_org = n_org
        self.n_rep = n_rep
        self.device = device if device else 'cpu'

        self.labels = torch.arange(n_org).to(self.device)
        self.mask = (1 - torch.eye(n_org)) > 0.
        self.cross_entropy = nn.CrossEntropyLoss()

    def drop_diag(self, x):
        x = x[self.mask]
        return x.reshape(self.n_org, self.n_org - 1)

    def forward(self, emb_org, emb_aug):
        ha, hb = emb_org, emb_aug
        logits_aa = (ha @ ha.T) / self.T
        logits_aa = self.drop_diag(logits_aa)
        logits_bb = (hb @ hb.T) / self.T
        logits_bb = self.drop_diag(logits_bb)
        logits_ab = (ha @ hb.T) / self.T
        logits_ba = (hb @ ha.T) / self.T
        loss_a = self.cross_entropy(torch.concatenate((logits_ab, logits_aa), dim=-1), self.labels)
        loss_b = self.cross_entropy(torch.concatenate((logits_ba, logits_bb), dim=-1), self.labels)
        return (loss_a + loss_b) / 2


class Focal_NTxent_Loss(nn.Module):

    def __init__(self, n_org: int, n_rep: int, T: float = 0.05, device: str = None, gamma: float = 1.):
        super(Focal_NTxent_Loss, self).__init__()
        self.T = T
        self.n_org = n_org
        self.n_rep = n_rep
        self.device = device if device else 'cpu'
        self.gamma = gamma

        self.labels = torch.arange(n_org).to(self.device)
        self.mask = (1 - torch.eye(n_org)) > 0.
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()

    def drop_diag(self, x):
        x = x[self.mask]
        return x.reshape(self.n_org, self.n_org - 1)

    def forward(self, emb_org, emb_aug):
        ha, hb = emb_org, emb_aug
        logits_aa = (ha @ ha.T) / self.T
        logits_aa = self.drop_diag(logits_aa)
        logits_bb = (hb @ hb.T) / self.T
        logits_bb = self.drop_diag(logits_bb)
        logits_ab = (ha @ hb.T) / self.T
        logits_ba = (hb @ ha.T) / self.T
        probs_a = F.softmax(torch.concatenate((logits_ab, logits_aa), dim=-1), dim=1)
        probs_b = F.softmax(torch.concatenate((logits_ba, logits_bb), dim=-1), dim=1)

        loss_a = self.nll_loss(
            ((1 - probs_a)**self.gamma) * self.log_softmax(torch.concatenate((logits_ab, logits_aa), dim=-1)),
            self.labels
        )
        loss_b = self.nll_loss(
            ((1 - probs_b)**self.gamma) * self.log_softmax(torch.concatenate((logits_ba, logits_bb), dim=-1)),
            self.labels
        )

        return (loss_a + loss_b) / 2