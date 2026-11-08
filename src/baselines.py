import numpy as np
from scipy.fft import irfft2, rfft2
from skimage.restoration import (denoise_nl_means, denoise_tv_chambolle,
                                 richardson_lucy, wiener)

from common import gaussian_psf

try:
    import bm3d as _bm3d
    HAS_BM3D = True
except Exception:
    HAS_BM3D = False


def rl(y, sigma=2.0, iters=30):
    return richardson_lucy(y, gaussian_psf(sigma), num_iter=iters, clip=False)


def tv(y, weight=0.02):
    return denoise_tv_chambolle(y, weight=weight)


def wiener_deconv(y, sigma=2.0, balance=0.02):
    return wiener(y, gaussian_psf(sigma), balance=balance)


def nlm(y, h=1.0):
    """Variance-stabilized non-local means (Anscombe + inverse)."""
    ya = 2.0 * np.sqrt(np.clip(y, 0, None) + 3.0 / 8.0)
    za = denoise_nl_means(ya, h=h, fast_mode=True, patch_size=5,
                          patch_distance=6)
    return ((za / 2.0) ** 2 - 3.0 / 8.0).astype(np.float32)


def _psf_fft(shape, sigma, n=15):
    k = gaussian_psf(sigma, n)
    K = np.zeros(shape, dtype=np.float32)
    K[:n, :n] = k
    K = np.roll(K, (-(n // 2), -(n // 2)), axis=(0, 1))
    return rfft2(K)


def tv_deconv(y, sigma=2.0, weight=0.05, rho=1.0, iters=40, pad=32):
    """ADMM TV-regularized deconvolution with the known Gaussian PSF.

    The forward model uses reflect boundaries (`common.blur`), so the ADMM
    runs on a reflect-padded image and crops back; this avoids the circular-
    convolution boundary penalty.
    """
    if pad:
        yp = np.pad(y, ((pad, pad), (pad, pad)), mode="reflect")
    else:
        yp = y.astype(np.float32)
    H = _psf_fft(yp.shape, sigma)
    Ht = np.conj(H)
    denom = np.abs(H) ** 2 + rho
    x = yp.astype(np.float32).copy()
    z = x.copy()
    u = np.zeros_like(x)
    for _ in range(iters):
        rhs = Ht * rfft2(yp) + rho * rfft2(z - u)
        x = irfft2(rhs / denom, s=yp.shape).astype(np.float32)
        z = denoise_tv_chambolle(x + u, weight=weight / rho)
        u = u + x - z
    return z[pad:-pad, pad:-pad] if pad else z


def rl_tv(y, sigma=2.0, iters=30, tv_weight=0.02):
    return denoise_tv_chambolle(rl(y, sigma, iters), weight=tv_weight)


def bm3d_anscombe(y, sigma_psd=1.0):
    if not HAS_BM3D:
        return y
    ya = 2.0 * np.sqrt(np.clip(y, 0, None) + 3.0 / 8.0)
    za = _bm3d.bm3d(ya.astype(np.float64), sigma_psd)
    return ((za / 2.0) ** 2 - 3.0 / 8.0).astype(np.float32)
