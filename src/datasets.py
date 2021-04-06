import torch

from torch.utils.data import Dataset
from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

import numpy as np


class DlibDataset(Dataset):
    def __init__(self, name, seed=0):
        self.dataset = get_named_ground_truth_data(name)
        self.random_state = np.random.RandomState(seed)
        self.len = self.dataset.images.shape[0]
        self.C = 1 if len(self.dataset.images.shape) == 3 else 3

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        x, y = self.sample(1)
        return x[0], y[0]
    
    def sample_factors(self, n):
        y = self.dataset.sample_factors(n, self.random_state)
        return torch.from_numpy(y)
    
    def sample_observations_from_factors(self, factors):
        y = factors.numpy()
        x = self.dataset.sample_observations_from_factors(y, self.random_state)
        x = torch.from_numpy(np.moveaxis(x, 3, 1))
        return x.to(torch.float32)
    
    def sample(self, n):
        y = self.sample_factors(n)
        x = self.sample_observations_from_factors(y)
        return x, y