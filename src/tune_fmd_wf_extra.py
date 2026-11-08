"""Extend the WF BM3D grid beyond 0.3 on the same validation fields."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
SIGMAS = [3.0, 5.0, 8.0, 12.0, 20.0]


def work(task):
    from baselines import bm3d_anscombe
    from fmd_common import load_pair
    from skimage.metrics import peak_signal_noise_ratio as psnr
    p, s = task
    n, g = load_pair(p[0], p[1])
    return s, float(psnr(g, np.clip(bm3d_anscombe(n, s), 0, 1),
                          data_range=1))


def main():
    from fmd_common import pairs
    wf = [p for p in pairs("val") if p[2].startswith("WF_")]
    wf = sorted(wf, key=lambda p: p[2])[:5]
    tasks = [(p, s) for s in SIGMAS for p in wf]
    with Pool(5, maxtasksperchild=1) as pool:
        res = pool.map(work, tasks)
    sc = {}
    for s, v in res:
        sc.setdefault(s, []).append(v)
    means = {s: float(np.mean(v)) for s, v in sc.items()}
    print(means, flush=True)
    best = max(means, key=means.get)
    p = Path("results/fmd_tuning_micro.json")
    d = json.load(open(p))
    if means[best] > means.get(3.0, -1):
        d["WF"]["bm3d"] = best
        p.write_text(json.dumps(d, indent=2))
        print("updated WF bm3d", best, flush=True)
    else:
        print("kept 0.3", flush=True)


if __name__ == "__main__":
    main()
