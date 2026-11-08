"""Physical widefield PSF check: Airy PSF instead of the training Gaussian."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.fft import irfft2, rfft2
from scipy.ndimage import convolve
from scipy.special import j1
from skimage.restoration import denoise_tv_chambolle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dataset_paths, load_image, patches, psnr, split_paths
from model import UNet

torch.set_num_threads(2)
DOSE = [3, 30, 300]
WG = [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2]
RHOS = [0.3, 1.0, 3.0]


def airy_psf(n=31, na=1.40, lam=0.52, pix=0.10):
    x = (np.arange(n) - n // 2) * pix
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx * xx + yy * yy)
    v = 2.0 * np.pi * na * r / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (2.0 * j1(v) / v) ** 2
    p[v == 0] = 1.0
    return (p / p.sum()).astype(np.float32)


def blur_psf(x, psf):
    return convolve(x.astype(np.float32), psf, mode="reflect")


def degrade_psf(x, psf, photons, rng):
    y = blur_psf(x, psf)
    return (rng.poisson(np.clip(y, 0, None) * photons) /
            photons).astype(np.float32)


def psf_fft(shape, psf):
    n0, n1 = psf.shape
    K = np.zeros(shape, dtype=np.float32)
    K[:n0, :n1] = psf
    K = np.roll(K, (-(n0 // 2), -(n1 // 2)), axis=(0, 1))
    return rfft2(K)


def tv_psf(y, psf, weight, rho, iters=30, pad=32):
    yp = np.pad(y, ((pad, pad), (pad, pad)), mode="reflect")
    H = psf_fft(yp.shape, psf)
    Ht = np.conj(H)
    den = np.abs(H) ** 2 + rho
    x = yp.astype(np.float32).copy()
    z = x.copy(); u = np.zeros_like(x)
    for _ in range(iters):
        rhs = Ht * rfft2(yp) + rho * rfft2(z - u)
        x = irfft2(rhs / den, s=yp.shape).astype(np.float32)
        z = denoise_tv_chambolle(x + u, weight=weight / rho)
        u = u + x - z
    return z[pad:-pad, pad:-pad]


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [g for _, _, g in ps[::step][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bbbc014_blind_strong.pt")
    ap.add_argument("--out", default="results/psf_airy.json")
    ap.add_argument("--params", default="")
    aa = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    psf = airy_psf()
    ck = torch.load(aa.ckpt, map_location=dev,
                    weights_only=False)
    net = UNet(ck.get("args", {}).get("ch", 64), ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    val = [load_image(p, scale) for p in sp["val"]][:5]
    test = [load_image(p, scale) for p in sp["test"]]
    vg = [g for img in val for g in sample(img)]
    tg = [sample(img) for img in test]
    if aa.params and Path(aa.params).exists():
        params = json.load(open(aa.params))["params"]
        print("loaded TV params from", aa.params, flush=True)
    else:
        params = {}
        for d in DOSE:
            rng = np.random.default_rng(2026)
            vy = [degrade_psf(g, psf, d, rng) for g in vg]
            best = (-1, None)
            for w in WG:
                for rho in RHOS:
                    m = float(np.mean([psnr(g, tv_psf(y, psf, w, rho))
                                       for g, y in zip(vg, vy)]))
                    if m > best[0]:
                        best = (m, (w, rho))
            params[str(d)] = dict(weight=best[1][0], rho=best[1][1],
                                  val_psnr=best[0])
            print("dose", d, params[str(d)], flush=True)
    out = {"psf": "Airy NA=1.4 lambda=520nm pixel=100nm", "params": params,
           "rows": {}}
    for d in DOSE:
        p = params[str(d)]
        rows = []
        for fi in range(len(test)):
            rng = np.random.default_rng(2026)
            ys = [degrade_psf(g, psf, d, rng) for g in tg[fi]]
            with torch.no_grad():
                u = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                        ).cpu().numpy()[:, 0]
            t = [tv_psf(y, psf, p["weight"], p["rho"]) for y in ys]
            rows.append(dict(
                unet=float(np.mean([psnr(g, q) for g, q in zip(tg[fi], u)])),
                tv=float(np.mean([psnr(g, q) for g, q in zip(tg[fi], t)]))))
        u = np.array([r["unet"] for r in rows])
        t = np.array([r["tv"] for r in rows])
        out["rows"][str(d)] = rows
        print(d, "U", round(u.mean(), 2), "TV", round(t.mean(), 2),
              "gain", round(float((u - t).mean()), 2),
              "wins", int((u > t).sum()), flush=True)
    Path(aa.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
