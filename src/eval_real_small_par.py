"""Parallel fair check on held-out small AI4Life pairs (fixed val-tuned params)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TUNED = {"bm3d": 0.5, "nlm": 0.5, "tv": 0.08}


def work(item):
    import torch
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    from baselines import bm3d_anscombe, nlm, tv
    from model import UNet
    from real_common import load_pair

    torch.set_num_threads(1)
    npath, gpath, stem = item
    n, g = load_pair(npath, gpath)
    dev = "cpu"
    ck = torch.load("checkpoints/real_unet.pt", map_location=dev,
                    weights_only=False)
    net = UNet(32, 1).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        p = net(torch.as_tensor(n)[None, None].to(dev)).cpu().numpy()[0, 0]
    preds = dict(noisy=n, bm3d=bm3d_anscombe(n, TUNED["bm3d"]),
                 nlm=nlm(n, TUNED["nlm"]), tv=tv(n, TUNED["tv"]), unet=p)
    out = {}
    for k, im in preds.items():
        out[k] = [float(psnr(g, np.clip(im, 0, 1), data_range=1)),
                  float(ssim(g, np.clip(im, 0, 1), data_range=1))]
    return int(stem), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--params", default="results/real_small_tuning.json")
    ap.add_argument("--out", default="results/real_small_test.json")
    a = ap.parse_args()

    global TUNED
    if a.params and Path(a.params).exists():
        TUNED = json.load(open(a.params))
    print("tuned params", TUNED, flush=True)

    from real_common import split_small
    _, _, test = split_small(0)
    if a.n:
        test = test[:a.n]
    print("test pairs", len(test), "workers", a.workers, flush=True)
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(work, test)
    print("elapsed", round(time.time() - t0, 1), flush=True)

    rows = [r for _, r in res]
    summary = {}
    for k in rows[0]:
        summary[k] = [float(np.mean([r[k][0] for r in rows])),
                      float(np.mean([r[k][1] for r in rows]))]
        print(k, "PSNR", round(summary[k][0], 2), "SSIM",
              round(summary[k][1], 3), flush=True)
    from scipy.stats import wilcoxon
    stats = {}
    for k in ["bm3d", "nlm", "tv"]:
        d = [r["unet"][0] - r[k][0] for r in rows]
        w = wilcoxon(d)
        stats[k] = dict(mean=float(np.mean(d)), wins=int(sum(x > 0 for x in d)),
                        n=len(d), p=float(w.pvalue))
        print(f"unet-{k}: {np.mean(d):+.2f} dB, wins "
              f"{sum(x>0 for x in d)}/{len(d)}, p={w.pvalue:.2e}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tuned=TUNED, rows=rows, summary=summary, paired=stats),
              open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
