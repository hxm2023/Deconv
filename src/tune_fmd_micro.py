"""Per-microscope FMD baseline tuning on validation fields."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
SIGMAS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]
NLM_H = [0.03, 0.05, 0.08, 0.1, 0.15]
TV_W = [0.02, 0.04, 0.06, 0.08, 0.12]


def work(task):
    from baselines import bm3d_anscombe
    from fmd_common import load_pair
    from skimage.metrics import peak_signal_noise_ratio as psnr

    npath, gpath, stem, sigma = task
    n, g = load_pair(npath, gpath)
    return stem, sigma, float(psnr(g, np.clip(bm3d_anscombe(n, sigma), 0, 1),
                                   data_range=1))


def main():
    from fmd_common import pairs
    val = pairs("val")
    groups = defaultdict(list)
    for p in val:
        groups[p[2].split("_")[0]].append(p)
    chosen = {m: sorted(v, key=lambda p: p[2])[:5] for m, v in groups.items()}
    tasks = [(p[0], p[1], p[2], s) for m in chosen for s in SIGMAS
             for p in chosen[m]]
    print("tasks", len(tasks), flush=True)
    with Pool(8, maxtasksperchild=1) as pool:
        res = pool.map(work, tasks)
    scores = defaultdict(lambda: defaultdict(list))
    for stem, sigma, v in res:
        scores[stem.split("_")[0]][sigma].append(v)
    best = {}
    for m in chosen:
        best[m] = dict(bm3d=max(SIGMAS, key=lambda s: np.mean(scores[m][s])))
        print(m, best[m], {s: round(float(np.mean(scores[m][s])), 2)
                           for s in SIGMAS}, flush=True)
    # NLM and TV are cheap; tune sequentially on the same fields
    from baselines import nlm, tv
    from fmd_common import load_pair
    from skimage.metrics import peak_signal_noise_ratio as psnr
    cache = {p[2]: load_pair(p[0], p[1]) for m in chosen for p in chosen[m]}
    for m in chosen:
        vals = [cache[p[2]] for p in chosen[m]]
        for name, grid in [("nlm", NLM_H), ("tv", TV_W)]:
            sc = {}
            for q in grid:
                ps = []
                for n, g in vals:
                    pred = nlm(n, q) if name == "nlm" else tv(n, q)
                    ps.append(psnr(g, np.clip(pred, 0, 1), data_range=1))
                sc[q] = float(np.mean(ps))
            best[m][name] = max(grid, key=lambda q: sc[q])
            print(m, name, best[m][name], sc, flush=True)
    Path("results/fmd_tuning_micro.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
