import torch
import numpy as np

import utils

class Collate_Fn():
    def __init__(self, rng: np.random.Generator, p:float = 0.33):
        self.rng = rng
        self.prob = p

    def __call__(self, batch):
        if self.rng.random() <= self.prob:
            mask = torch.from_numpy(utils.cutout_spec_augment_mask(self.rng))
            x_orgs = [mask * sample[0] for sample in batch]
            x_augs = [mask * sample[1] for sample in batch]
            return torch.stack(x_orgs), torch.stack(x_augs)
        else:
            x_orgs, x_augs = list(zip(*batch))
            return torch.stack(x_orgs), torch.stack(x_augs)