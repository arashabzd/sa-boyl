import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
        )
    
    def forward(self, x):
        out = self.projector(x)
        return out

class Predictor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(num_features, num_features, bias=False),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features)
        )
    
    def forward(self, x):
        out = self.predictor(x)
        return out