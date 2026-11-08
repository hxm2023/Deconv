"""Extend the dose range to P=1 and check for a larger fair-TV margin."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv_deconv
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, ssim, split_paths)
from model import UNet

torch.set_num_threads(2)
LAMS = [0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8]


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def tune_tv(vg, dose):
    rng = np.random.default_rng(2026)
    vy = [degrade(g, 2.0, dose, rng) for g in vg]
    best = (-1, None)
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
    return best3[1], rho, it, best3[0]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_blind_p1_strong.pt", map_location=dev,
                    weights_only=False)
    ch = ck.get("args", {}).get("ch", 64)
    net = UNet(ch, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    val = [load_image(p, scale) for p in sp["val"]][:5]
    test = [load_image(p, scale) for p in sp["test"]]
    vg = [g for img in val for g in sample(img)]
    tg = [sample(img) for img in test]
    out = {}
    for dose in [1, 3, 30, 300]:
        lam, rho, it, vpsnr = tune_tv(vg, dose)
        rows = []
        for fi in range(len(test)):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in tg[fi]]
            with torch.no_grad():
                p = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                        ).cpu().numpy()[:, 0]
            p_tv = [tv_deconv(y, 2.0, lam, rho=rho, iters=it, pad=32)
                    for y in ys]
            rows.append(dict(
                unet=float(np.mean([psnr(g, q) for g, q in zip(tg[fi], p)])),
                tv=float(np.mean([psnr(g, q) for g, q in zip(tg[fi], p_tv)])),
                unet_ssim=float(np.mean([ssim(g, q)
                                         for g, q in zip(tg[fi], p)])),
                tv_ssim=float(np.mean([ssim(g, q)
                                       for g, q in zip(tg[fi], p_tv)])),
                unet_fg=float(np.mean([fg_psnr(g, q)
                                       for g, q in zip(tg[fi], p)])),
                tv_fg=float(np.mean([fg_psnr(g, q)
                                     for g, q in zip(tg[fi], p_tv)]))))
        u = np.array([r["unet"] for r in rows])
        t = np.array([r["tv"] for r in rows])
        out[str(dose)] = dict(lam=lam, rho=rho, iters=it, val_psnr=vpsnr,
                              rows=rows, unet_mean=float(u.mean()),
                              tv_mean=float(t.mean()),
                              gain=float((u - t).mean()),
                              wins=int((u > t).sum()))
        print(dose, "lam", lam, "rho", rho, "it", it,
              "U", round(u.mean(), 2), "TV", round(t.mean(), 2),
              "gain", round(float((u - t).mean()), 2),
              "wins", int((u > t).sum()), flush=True)
    Path("results/p1_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
