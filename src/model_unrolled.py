"""Physics-informed unrolled deconvolution: learned proximal steps with the PSF."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import gaussian_psf


class UnrolledDeconv(nn.Module):
    def __init__(self, ch=32, K=5, sigma=2.0):
        super().__init__()
        self.K = K
        k = torch.as_tensor(gaussian_psf(sigma, 15))[None, None]
        self.register_buffer("psf", k)
        self.eta_raw = nn.Parameter(torch.full((K,), 0.0))
        self.prox = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1))

    def _H(self, x):
        return F.conv2d(x, self.psf, padding=7)

    def _Ht(self, x):
        return F.conv2d(x, self.psf.flip(-1, -2), padding=7)

    def forward(self, x):
        y = x[:, :1]
        c = x[:, 1:2]
        z = y
        for k in range(self.K):
            eta = F.softplus(self.eta_raw[k])
            g = z - eta * self._Ht(self._H(z) - y)
            z = g + self.prox(torch.cat([g, y, c], 1))
        return z
