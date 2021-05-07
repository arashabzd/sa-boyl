import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, c_dim, z_dim, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(c_dim+z_dim, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 1024, bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        
        self.conv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, z):
        out = torch.cat([c, z], dim=1)
        out = self.relu(self.bn1(self.linear1(out)))
        out = self.relu(self.bn2(self.linear2(out)))
        out = out.view(-1, 64, 4, 4)
        out = self.lrelu(self.bn3(self.conv1(out)))
        out = self.lrelu(self.bn4(self.conv2(out)))
        out = self.lrelu(self.bn5(self.conv3(out)))
        out = self.sigmoid(self.conv4(out))
        return out