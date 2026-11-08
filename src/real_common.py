"""AI4Life/MDC25 real nuclei noisy-clean pairs."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from skimage import io

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "real_nuclei"


def pairs(split="small"):
    gt = sorted(glob.glob(str(DATA / f"{split}_images" / "gt" / "*.tif")))
    out = []
    for g in gt:
        stem = os.path.basename(g)
        n = DATA / f"{split}_images" / "noisy" / stem
        if n.exists():
            out.append((str(n), str(g), int(Path(stem).stem)))
    return out


def norm99(x):
    lo, hi = np.percentile(x, [0.5, 99.9])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def load_pair(npath, gpath):
    n = io.imread(npath).astype(np.float32)
    g = io.imread(gpath).astype(np.float32)
    return norm99(n), norm99(g)


def split_small(seed=0):
    ps = pairs("small")
    # exclude crops of the large held-out fields 1000-1019
    ps = [p for p in ps if p[2] > 1019]
    ids = sorted(set(p[2] for p in ps))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    n_val = max(1, len(ids) // 10)
    val = set(int(x) for x in perm[:n_val])
    test = set(int(x) for x in perm[n_val:2 * n_val])
    train = [p for p in ps if p[2] not in val and p[2] not in test]
    valp = [p for p in ps if p[2] in val]
    testp = [p for p in ps if p[2] in test]
    return train, valp, testp
