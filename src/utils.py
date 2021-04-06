import torch

import numpy as np


def create_mask(n, d, repeat=1, fill=0):
    mask = torch.empty(n, d).fill_(fill)
    idx = torch.randint(d, (n,))
    mask[torch.arange(n), idx] = 1 - fill
    mask = torch.repeat_interleave(mask, repeat, dim=1)
    return mask

def create_factors(n, d, g=0, repeat=1, fill=0):
    c = torch.empty(n, d*repeat).uniform_(-1, 1)
    mask = create_mask(n, d, repeat=repeat, fill=fill)
    c1 = torch.empty(n, d*repeat).uniform_(-1 + g/2, 1 - g/2)
    c2 = torch.empty(n, d*repeat).uniform_(-1 + g/2, 1 - g/2)
    c1_ = (c1 > c2).to(torch.float32) * (c1 + g/2) + (c1 <= c2).to(torch.float32) * (c1 - g/2)
    c2_ = (c1 > c2).to(torch.float32) * (c2 - g/2) + (c1 <= c2).to(torch.float32) * (c2 + g/2)
    c1 = mask * c + (1 - mask) * c1_
    c2 = mask * c + (1 - mask) * c2_
    return c1, c2, mask

def make_representor(encoder, device):
    encoder = encoder.to(device)
    encoder.eval()
    def represent(x):
        x = np.moveaxis(x, 3, 1)
        x = torch.from_numpy(x).to(torch.float32)
        with torch.no_grad():
            z = encoder(x.to(device))
        return z.cpu().numpy()
    return represent

