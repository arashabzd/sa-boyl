import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=.2, inplace=True),
            spectral_norm(nn.Conv2d(32, 32, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=.2, inplace=True),
            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=.2, inplace=True),
            nn.Flatten(),
            spectral_norm(nn.Linear(64*4*4, 128)),
            nn.LeakyReLU(negative_slope=.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out