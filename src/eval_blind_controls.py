"""Evaluate the mixed-dose blind model as the main method, plus controls."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, ssim)
from model import UNet


def predict(net, ys, dev, dose=None, dose_cond=False, batch=64):
    out = []
    import math
    c = None
    if dose_cond:
        c = (math.log10(dose) - math.log10(3.0)) / (
            math.log10(300.0) - math.log10(3.0))
    with torch.no_grad():
        for q in range(0, len(ys), batch):
            x = torch.as_tensor(np.asarray(ys[q:q + batch]),
                                dtype=torch.float32)[:, None].to(dev)
            if dose_cond:
                ch = torch.full((len(x), 1, x.shape[2], x.shape[3]), c,
                                device=dev)
                x = torch.cat([x, ch], 1)
            out.append(net(x).cpu().numpy()[:, 0])
    return np.concatenate(out, 0)


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    ch = ck.get("args", {}).get("ch", 32)
    net = UNet(ch, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths, scale = dataset_paths("bbbc014")
    from common import split_paths
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    gts = []
    for img in test:
        ps = list(patches(img, 128))
        step = max(1, len(ps) // 10)
        gts.append([g for _, _, g in ps[::step][:10]])

    net, _ = load("checkpoints/bbbc014_blind_strong.pt", dev)
    out = {}
    for dose in [3, 10, 30, 100, 300]:
        rows = []
        for fi, img in enumerate(test):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in gts[fi]]
            p = predict(net, ys, dev)
            rows.append(dict(
                psnr=float(np.mean([psnr(g, q) for g, q in zip(gts[fi], p)])),
                ssim=float(np.mean([ssim(g, q) for g, q in zip(gts[fi], p)])),
                fg=float(np.mean([fg_psnr(g, q) for g, q in zip(gts[fi], p)]))))
        out[str(dose)] = rows
        print("blind", dose, {k: round(float(np.mean([r[k] for r in rows])), 3)
                              for k in rows[0]}, flush=True)
    json.dump(out, open("results/blind_fair_doses.json", "w"), indent=2)

    # equal-budget scrambled control
    scr, sck = load("checkpoints/bbbc014_scr2.pt", dev)
    sout = {}
    for dose in [3, 30, 100]:
        rows = []
        for fi, img in enumerate(test):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in gts[fi]]
            p = predict(scr, ys, dev, dose=dose, dose_cond=sck["dose_cond"])
            rows.append(float(np.mean([psnr(g, q) for g, q in zip(gts[fi], p)])))
        sout[str(dose)] = rows
        print("scr2", dose, round(float(np.mean(rows)), 2), flush=True)
    json.dump(sout, open("results/scr2_matched.json", "w"), indent=2)

    # external zero-shot, blind model
    eout = {}
    for key in ["bbbc015", "bbbc016", "bbbc018"]:
        ep, es = dataset_paths(key)
        ep = ep[::max(1, len(ep) // 20)][:20]
        rows = []
        for pth in ep:
            img = load_image(pth, es)
            ps = list(patches(img, 128))
            step = max(1, len(ps) // 10)
            gg = [g for _, _, g in ps[::step][:10]]
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, 30.0, rng) for g in gg]
            p = predict(net, ys, dev)
            rows.append(float(np.mean([psnr(g, q) for g, q in zip(gg, p)])))
        eout[key] = rows
        print(key, round(float(np.mean(rows)), 2), flush=True)
    json.dump(eout, open("results/blind_external30.json", "w"), indent=2)

    # PSF robustness
    pout = {}
    for sigma in [1.5, 2.0, 2.5]:
        rows = []
        for fi in range(len(test)):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, sigma, 30.0, rng) for g in gts[fi]]
            p = predict(net, ys, dev)
            rows.append(float(np.mean([psnr(g, q) for g, q in zip(gts[fi], p)])))
        pout[str(sigma)] = rows
        print("sigma", sigma, round(float(np.mean(rows)), 2), flush=True)
    json.dump(pout, open("results/blind_sigma.json", "w"), indent=2)


if __name__ == "__main__":
    main()
