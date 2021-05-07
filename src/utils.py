import torch

import numpy as np


def mask_like(x, fill=0):
    n, d = x.shape
    m = torch.empty_like(x).fill_(fill)
    idx = torch.randint(d, (n,))
    m[torch.arange(n), idx] = 1 - fill
    return m

def consistent_like(c):
    m_con = mask_like(c, fill=0)
    noise = torch.empty_like(c).uniform_(-1, 1)
    c_con = m_con * c + (1 - m_con) * noise
    return c_con, m_con

def restrictive_like(c):
    m_res = mask_like(c, fill=1)
    noise = torch.empty_like(c).uniform_(-1, 1)
    c_res = m_res * c + (1 - m_res) * noise
    return c_res, m_res

def representor(encoder, device):
    encoder = encoder.to(device)
    encoder.eval()
    def represent(x):
        x = np.moveaxis(x, 3, 1)
        x = torch.from_numpy(x).to(torch.float32)
        with torch.no_grad():
            z = encoder(x.to(device))
        return z.cpu().numpy()
    return represent

