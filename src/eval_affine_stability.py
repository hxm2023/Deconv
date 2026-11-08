"""Stability of the real-data affine calibration on the validation split."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
TUNED = {"bm3d": 0.3, "nlm": 0.2, "tv": 0.24}


def work(item):
    from baselines import bm3d_anscombe, nlm, tv
    from real_common import load_pair

    npath, gpath, stem = item
    n, g = load_pair(npath, gpath)
    y = g.ravel().astype(np.float64)
    out = {"stem": int(stem), "n": int(y.size), "sy": float(y.sum()),
           "syy": float((y * y).sum())}
    for name, pred in [("bm3d", bm3d_anscombe(n, TUNED["bm3d"])),
                       ("nlm", nlm(n, TUNED["nlm"])),
                       ("tv", tv(n, TUNED["tv"]))]:
        x = pred.ravel().astype(np.float64)
        out[name] = dict(sx=float(x.sum()), sxx=float((x * x).sum()),
                         sxy=float((x * y).sum()))
    return out


def fit(stats):
    y = np.array([s["sy"] for s in stats])
    n = np.array([s["n"] for s in stats])
    sy = y.sum(); syy = sum(s["syy"] for s in stats); N = n.sum()
    out = {}
    for m in ["bm3d", "nlm", "tv"]:
        sx = sum(s[m]["sx"] for s in stats)
        sxx = sum(s[m]["sxx"] for s in stats)
        sxy = sum(s[m]["sxy"] for s in stats)
        den = N * sxx - sx * sx
        a = (N * sxy - sx * sy) / den
        b = (sy - a * sx) / N
        out[m] = [float(a), float(b)]
    return out


def main():
    from real_common import split_small
    _, val, _ = split_small(0)
    print("val pairs", len(val), flush=True)
    with Pool(8) as pool:
        stats = pool.map(work, val)
    out = {}
    for k in [5, 20, 50, 243]:
        out[str(k)] = fit(stats[:k])
        print(k, out[str(k)], flush=True)
    out["order"] = [s["stem"] for s in stats]
    Path("results/real_affine_stability.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
