"""Check true TV deconvolution against denoising-only and the U-Net."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import rl, rl_tv, tv, tv_deconv
from common import dataset_paths, degrade, load_image, patches, psnr, split_paths, ssim
from model import UNet


def unet_predict(net, y, dose, dev):
    import math
    c = (math.log10(dose) - math.log10(3.0)) / (math.log10(300.0) - math.log10(3.0))
    x = torch.as_tensor(y, dtype=torch.float32)[None, None].to(dev)
    ch = torch.full((1, 1, y.shape[0], y.shape[1]), c, device=dev)
    with torch.no_grad():
        return net(torch.cat([x, ch], 1)).cpu().numpy()[0, 0]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_dcond.pt", map_location=dev,
                    weights_only=False)
    net = UNet(32, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    val = [load_image(p, scale) for p in sp["val"]]
    test = [load_image(p, scale) for p in sp["test"]]

    def items(fovs, n=5):
        out = []
        for img in fovs:
            ps = list(patches(img, 128))
            step = max(1, len(ps) // n)
            out += [gt for _, _, gt in ps[::step][:n]]
        return out
    vi = items(val, 3)
    ti = items(test, 5)

    for dose in [3.0, 30.0]:
        rng = np.random.default_rng(2026)
        vy = [degrade(gt, 2.0, dose, rng) for gt in vi]
        best = (-1, None)
        for lam in [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2]:
            t0 = time.time()
            vals = [psnr(gt, tv_deconv(y, 2.0, lam, rho=1.0, iters=30))
                    for gt, y in zip(vi, vy)]
            m = float(np.mean(vals))
            print(f"dose {dose} lam {lam}: val {m:.2f} ({time.time()-t0:.1f}s)",
                  flush=True)
            if m > best[0]:
                best = (m, lam)
        lam = best[1]
        print("best", lam, best[0], flush=True)
        rng = np.random.default_rng(2026)
        ty = [degrade(gt, 2.0, dose, rng) for gt in ti]
        methods = {}
        t0 = time.time()
        methods["tv_deconv"] = [tv_deconv(y, 2.0, lam, iters=30) for y in ty]
        print("test tv_deconv", round(time.time() - t0, 1), "s", flush=True)
        methods["tv_denoise"] = [tv(y, 0.4 if dose == 3 else 0.12) for y in ty]
        methods["rl"] = [rl(y, 2.0, 30) for y in ty]
        methods["rl_tv"] = [rl_tv(y, 2.0, 30, 0.02) for y in ty]
        methods["unet"] = [unet_predict(net, y, dose, dev) for y in ty]
        print("dose", dose)
        for m, ps in methods.items():
            print(f"  {m:10s} PSNR {np.mean([psnr(g, p) for g, p in zip(ti, ps)]):.2f} "
                  f"SSIM {np.mean([ssim(g, p) for g, p in zip(ti, ps)]):.4f}")


if __name__ == "__main__":
    main()
