"""Fair deconvolution baseline: ADMM TV with the known PSF, versus U-Net."""
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
from baselines import tv, tv_deconv
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, save_json, split_paths, ssim)
from model import UNet

DOSES = [3, 10, 30, 100, 300]
LAMS = [0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3]
N_PATCH = 10


def sample(img, n=N_PATCH):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def unet_predict(net, ys, dose, dev, batch=64):
    import math
    c = (math.log10(dose) - math.log10(3.0)) / (math.log10(300.0) - math.log10(3.0))
    out = []
    with torch.no_grad():
        for q in range(0, len(ys), batch):
            x = torch.as_tensor(np.asarray(ys[q:q + batch]), dtype=torch.float32)[:, None]
            ch = torch.full((len(x), 1, x.shape[2], x.shape[3]), c)
            out.append(net(torch.cat([x, ch], 1).to(dev)).cpu().numpy()[:, 0])
    return np.concatenate(out, 0)


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
    val_imgs = val[:5]
    out = {}
    for dose in DOSES:
        rng = np.random.default_rng(2026)
        vg = [g for img in val_imgs for g in sample(img)]
        vy = [degrade(g, 2.0, dose, rng) for g in vg]
        best = (-1, None)
        t0 = time.time()
        for lam in LAMS:
            m = float(np.mean([psnr(g, tv_deconv(y, 2.0, lam, rho=1.0,
                                                iters=30, pad=32))
                               for g, y in zip(vg, vy)]))
            if m > best[0]:
                best = (m, lam)
        lam = best[1]
        best2 = (-1, None)
        for rho in [0.3, 1.0, 3.0]:
            for it in [30, 60, 100]:
                m = float(np.mean([psnr(g, tv_deconv(y, 2.0, lam, rho=rho,
                                                    iters=it, pad=32))
                                   for g, y in zip(vg, vy)]))
                if m > best2[0]:
                    best2 = (m, (rho, it))
        rho, it = best2[1]
        best3 = (-1, None)
        for lam in LAMS:
            m = float(np.mean([psnr(g, tv_deconv(y, 2.0, lam, rho=rho,
                                                iters=it, pad=32))
                               for g, y in zip(vg, vy)]))
            if m > best3[0]:
                best3 = (m, lam)
        lam = best3[1]
        print(f"dose {dose}: tuned lam {lam} rho {rho} iters {it} val "
              f"{best3[0]:.2f} ({time.time()-t0:.0f}s)", flush=True)
        rows = []
        for img in test:
            gts = sample(img)
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in gts]
            p_tv = [tv_deconv(y, 2.0, lam, rho=rho, iters=it, pad=32)
                    for y in ys]
            p_dn = [tv(y, {3: 0.4, 10: 0.3, 30: 0.12, 100: 0.04, 300: 0.02}[dose])
                    for y in ys]
            p_un = unet_predict(net, ys, dose, dev)
            rows.append(dict(
                tv_deconv=[psnr(g, p) for g, p in zip(gts, p_tv)],
                tv_denoise=[psnr(g, p) for g, p in zip(gts, p_dn)],
                unet=[psnr(g, p) for g, p in zip(gts, p_un)],
                tv_deconv_ssim=[ssim(g, p) for g, p in zip(gts, p_tv)],
                tv_denoise_ssim=[ssim(g, p) for g, p in zip(gts, p_dn)],
                unet_ssim=[ssim(g, p) for g, p in zip(gts, p_un)],
                tv_deconv_fg=[fg_psnr(g, p) for g, p in zip(gts, p_tv)],
                tv_denoise_fg=[fg_psnr(g, p) for g, p in zip(gts, p_dn)],
                unet_fg=[fg_psnr(g, p) for g, p in zip(gts, p_un)],
            ))
        out[str(dose)] = dict(lam=lam, rho=rho, iters=it, rows=rows)
        print(f"  test means: TVdec {np.mean([np.mean(r['tv_deconv']) for r in rows]):.2f} "
              f"TVden {np.mean([np.mean(r['tv_denoise']) for r in rows]):.2f} "
              f"UNet {np.mean([np.mean(r['unet']) for r in rows]):.2f}", flush=True)
    save_json("results/tvdeconv_doses.json", out)


if __name__ == "__main__":
    main()
