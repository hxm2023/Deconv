"""Recompute real test metrics with affine maps fitted on all 243 val pairs."""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
TUNED = {"bm3d": 0.3, "nlm": 0.2, "tv": 0.24}
AFF = None


def work(item):
    from baselines import bm3d_anscombe, nlm, tv
    from real_common import load_pair
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    npath, gpath, stem = item
    n, g = load_pair(npath, gpath)
    out = {}
    for name, pred in [("bm3d", bm3d_anscombe(n, TUNED["bm3d"])),
                       ("nlm", nlm(n, TUNED["nlm"])),
                       ("tv", tv(n, TUNED["tv"]))]:
        a, b = AFF[name]
        q = np.clip(a * pred + b, 0, 1)
        out[name] = [float(psnr(g, q, data_range=1)),
                     float(ssim(g, q, data_range=1))]
    return str(stem), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=20)
    a = ap.parse_args()
    global AFF
    AFF = json.load(open("results/real_affine_stability.json"))["243"]
    base = json.load(open("results/real_affine_test_all_strong.json"))
    from real_common import split_small
    _, _, test = split_small(0)
    test = test[:a.n_test]
    keep = {str(t[2]) for t in test}
    print("tasks", len(test), flush=True)
    with Pool(8) as pool:
        res = dict(pool.map(work, test))
    rows = []
    for r in base["rows"]:
        if r["stem"] not in keep:
            continue
        rr = dict(r)
        for k in ["bm3d", "nlm", "tv"]:
            rr[k] = res[r["stem"]][k]
        rows.append(rr)
    summary = {}
    for k in ["unet", "bm3d", "nlm", "tv"]:
        summary[k] = [float(np.mean([r[k][0] for r in rows])),
                      float(np.mean([r[k][1] for r in rows]))]
    d = np.array([r["unet"][0] - r["bm3d"][0] for r in rows])
    from scipy.stats import wilcoxon
    print("unet", round(summary["unet"][0], 2),
          "bm3d", round(summary["bm3d"][0], 2),
          "gain", round(float(d.mean()), 2),
          "wins", int((d > 0).sum()),
          "p", float(wilcoxon(d).pvalue), flush=True)
    out = dict(tuned=TUNED, affine=AFF, rows=rows, summary=summary,
               ckpt=base.get("ckpt"))
    Path("results/real_affine_test_all_strong243.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
