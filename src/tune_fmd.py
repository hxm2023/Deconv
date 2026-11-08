"""Extended classical tuning on FMD validation pairs (parallel)."""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

GRID = {
    "bm3d": [0.10, 0.30, 0.80],
    "nlm": [0.10, 0.20],
    "tv": [0.08, 0.20],
}


def work(task):
    name, v, npath, gpath = task
    from skimage.metrics import peak_signal_noise_ratio as psnr

    from baselines import bm3d_anscombe, nlm, tv
    from fmd_common import load_pair

    fn = {"bm3d": lambda n: bm3d_anscombe(n, v),
          "nlm": lambda n: nlm(n, v),
          "tv": lambda n: tv(n, v)}[name]
    n, g = load_pair(npath, gpath)
    return name, v, float(psnr(g, np.clip(fn(n), 0, 1), data_range=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--per-group", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/fmd_tuning.json")
    a = ap.parse_args()
    from fmd_common import pairs
    val = pairs("val")
    if a.per_group:
        groups = {}
        for p in val:
            groups.setdefault(tuple(p[2].split("_")[:3]), []).append(p)
        val = [p for k in sorted(groups) for p in groups[k][:a.per_group]]
    if a.n_val:
        val = val[:a.n_val]
    tasks = [(name, v, p[0], p[1]) for name, grid in GRID.items()
             for v in grid for p in val]
    print("tasks", len(tasks), "val pairs", len(val), flush=True)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(work, tasks)
    scores = {}
    for name, v, s in res:
        scores.setdefault(name, {}).setdefault(v, []).append(s)
    best = {}
    for name, d in scores.items():
        means = {v: float(np.mean(s)) for v, s in d.items()}
        best[name] = max(means, key=means.get)
        print(name, {v: round(m, 3) for v, m in means.items()},
              "-> best", best[name], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(best, open(a.out, "w"), indent=2)
    grid_means = {name: {str(v): float(np.mean(s)) for v, s in d.items()}
                  for name, d in scores.items()}
    json.dump(grid_means, open(str(a.out).replace(".json", "_grid.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
