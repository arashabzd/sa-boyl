import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Predictor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.predictor = nn.Sequential(
            spectral_norm(nn.Linear(2*in_features, 128, bias=False)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, 128, bias=False)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, in_features))
        )
    
    def forward(self, e, c):
        out = torch.cat([e, c], dim=1)
        out = self.predictor(out)
        return out