"""Pre-experiment: headroom for blind aberration deconvolution.

Each patch gets a random Zernike aberration (defocus, astigmatism, coma). A
classical TV method with a nominal in-focus Gaussian PSF is compared with an
oracle that knows the true PSF. A large gap means a model that estimates the
aberration from the image could add a capability that classical deconvolution
cannot provide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.fft import irfft2, rfft2
from scipy.ndimage import convolve
from skimage.restoration import denoise_tv_chambolle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dataset_paths, degrade, gaussian_psf, load_image, patches, psnr, split_paths
from model import UNet


def zernike_psf(a4, a5, a6, n=64, size=31):
    y, x = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    z4 = 2 * r ** 2 - 1
    z5 = r ** 2 * np.cos(2 * th)
    z6 = (3 * r ** 3 - 2 * r) * np.cos(th)
    phase = 2 * np.pi * (a4 * z4 + a5 * z5 + a6 * z6)
    pupil = (r <= 1) * np.exp(1j * phase)
    psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil)))) ** 2
    c = n // 2
    h = size // 2
    psf = psf[c - h:c + h + 1, c - h:c + h + 1]
    return (psf / psf.sum()).astype(np.float32)


def blur_psf(x, psf):
    return convolve(x.astype(np.float32), psf, mode="reflect")


def tv_psf(y, psf, weight=0.04, rho=0.3, iters=30, pad=32):
    yp = np.pad(y, ((pad, pad), (pad, pad)), mode="reflect")
    K = np.zeros(yp.shape, dtype=np.float32)
    K[:psf.shape[0], :psf.shape[1]] = psf
    K = np.roll(K, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    H = rfft2(K)
    Ht = np.conj(H)
    denom = np.abs(H) ** 2 + rho
    x = yp.astype(np.float32).copy()
    z = x.copy(); u = np.zeros_like(x)
    for _ in range(iters):
        rhs = Ht * rfft2(yp) + rho * rfft2(z - u)
        x = irfft2(rhs / denom, s=yp.shape).astype(np.float32)
        z = denoise_tv_chambolle(x + u, weight=weight / rho)
        u = u + x - z
    return z[pad:-pad, pad:-pad]


def sample(img, n=5):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_blind3.pt", map_location=dev,
                    weights_only=False)
    net = UNet(32, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    rng = np.random.default_rng(2026)
    nominal_psf = gaussian_psf(2.0)
    rows = {"nominal": [], "oracle": [], "unet": []}
    for img in test:
        for g in sample(img):
            a4, a5, a6 = rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
            psf = zernike_psf(a4, a5, a6)
            y = rng.poisson(np.clip(blur_psf(g, psf), 0, None) * 30.0) / 30.0
            y = y.astype(np.float32)
            p_nom = tv_psf(y, nominal_psf)
            p_or = tv_psf(y, psf)
            with torch.no_grad():
                p_un = net(torch.as_tensor(y)[None, None].to(dev)).cpu().numpy()[0, 0]
            rows["nominal"].append(psnr(g, p_nom))
            rows["oracle"].append(psnr(g, p_or))
            rows["unet"].append(psnr(g, p_un))
    out = {k: float(np.mean(v)) for k, v in rows.items()}
    out["headroom_tv"] = out["oracle"] - out["nominal"]
    out["headroom_unet"] = out["oracle"] - out["unet"]
    out["n"] = len(rows["nominal"])
    print(out, flush=True)
    Path("results/aberration_headroom.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
