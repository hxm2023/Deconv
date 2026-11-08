"""Parallel evaluation on the field-split FMD test set (fixed val-tuned params)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TUNED = {"bm3d": 0.10, "nlm": 0.10, "tv": 0.08}
MICRO = None


def work(item):
    import torch
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    from baselines import bm3d_anscombe, nlm, tv
    from fmd_common import load_pair
    from model import UNet

    torch.set_num_threads(1)
    npath, gpath, stem = item
    n, g = load_pair(npath, gpath)
    dev = "cpu"
    ck = torch.load("checkpoints/fmd_unet.pt", map_location=dev,
                    weights_only=False)
    net = UNet(32, 1).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        p = net(torch.as_tensor(n)[None, None].to(dev)).cpu().numpy()[0, 0]
    q = MICRO[stem.split("_")[0]] if MICRO else TUNED
    preds = dict(noisy=n, bm3d=bm3d_anscombe(n, q["bm3d"]),
                 nlm=nlm(n, q["nlm"]), tv=tv(n, q["tv"]), unet=p)
    out = {}
    for k, im in preds.items():
        out[k] = [float(psnr(g, np.clip(im, 0, 1), data_range=1)),
                  float(ssim(g, np.clip(im, 0, 1), data_range=1))]
    return stem, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--per-group", type=int, default=0,
                    help="sample this many test fields per microscope/channel/level")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--params", default="results/fmd_tuning.json")
    ap.add_argument("--out", default="results/fmd_test.json")
    ap.add_argument("--subset", default="")
    ap.add_argument("--params-micro", default="")
    a = ap.parse_args()
    global TUNED, MICRO
    if a.params and Path(a.params).exists():
        TUNED = json.load(open(a.params))
    if a.params_micro and Path(a.params_micro).exists():
        MICRO = json.load(open(a.params_micro))
    print("tuned params", TUNED, flush=True)
    from fmd_common import pairs
    test = pairs("test")
    if a.subset:
        keep = set(json.load(open(a.subset))["stems"])
        test = [p for p in test if p[2] in keep]
    if a.per_group:
        groups = {}
        for p in test:
            groups.setdefault(tuple(p[2].split("_")[:3]), []).append(p)
        test = [p for k in sorted(groups) for p in groups[k][:a.per_group]]
    if a.n:
        test = test[:a.n]
    print("test pairs", len(test), "workers", a.workers, flush=True)
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(work, test)
    print("elapsed", round(time.time() - t0, 1), flush=True)
    rows = []
    for stem, r in res:
        parts = stem.split("_")
        r.update(stem=stem, microscope=parts[0], channel=parts[1],
                 level=parts[2])
        rows.append(r)
    summary = {}
    for k in ["noisy", "bm3d", "nlm", "tv", "unet"]:
        summary[k] = [float(np.mean([r[k][0] for r in rows])),
                      float(np.mean([r[k][1] for r in rows]))]
        print(k, "PSNR", round(summary[k][0], 2), "SSIM",
              round(summary[k][1], 3), flush=True)
    from scipy.stats import wilcoxon
    stats = {}
    for k in ["bm3d", "nlm", "tv"]:
        d = [r["unet"][0] - r[k][0] for r in rows]
        w = wilcoxon(d)
        stats[k] = dict(mean=float(np.mean(d)),
                        wins=int(sum(x > 0 for x in d)), n=len(d),
                        p=float(w.pvalue))
        print(f"unet-{k}: {np.mean(d):+.2f} dB, wins "
              f"{sum(x>0 for x in d)}/{len(d)}, p={w.pvalue:.2e}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    scope = ("per-microscope; tuned on five validation fields per microscope"
             if MICRO else "global; tuned on five validation fields")
    json.dump(dict(params_scope=scope, params_micro_used=MICRO,
                   tuned_global_legacy=TUNED, rows=rows, summary=summary,
                   paired=stats), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
