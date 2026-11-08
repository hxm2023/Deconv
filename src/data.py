"""Low-dose microscopy deconvolution data from public skimage images."""
import numpy as np
from skimage import data, color, transform
from scipy.ndimage import gaussian_filter


def load_images(size=128):
    imgs = {"cell": data.cell(), "retina": data.retina(),
            "ihc": data.immunohistochemistry(), "camera": data.camera(),
            "coins": data.coins(), "brick": data.brick()}
    out = {}
    for k, v in imgs.items():
        if v.ndim == 3:
            v = color.rgb2gray(v)
        v = transform.resize(v, (size, size), anti_aliasing=True)
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        out[k] = v.astype(np.float32)
    return out


def degrade(x, sigma=2.0, photons=300.0, rng=None):
    y = gaussian_filter(x, sigma, mode="reflect")
    y = rng.poisson(np.clip(y, 0, None) * photons) / photons
    return y.astype(np.float32)
