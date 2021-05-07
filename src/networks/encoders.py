import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Encoder(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(32, 32, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(64, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            spectral_norm(nn.Linear(64*4*4, 128, bias=False)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, out_dim)),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.encoder(x)
        return out