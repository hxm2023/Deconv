"""Median detected photons per pixel at P=3/30/300 on the test patches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import blur, dataset_paths, load_image, patches, split_paths

DOSES = [3, 30, 300]


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    out = {}
    for dose in DOSES:
        per_patch = []
        pooled = []
        for img in test:
            for g in sample(img):
                v = dose * blur(g)
                per_patch.append(float(np.median(v)))
                pooled.append(v.ravel())
        pooled = np.concatenate(pooled)
        out[str(dose)] = dict(
            mean_expected=float(pooled.mean()),
            mean_of_patch_medians=float(np.mean(per_patch)),
            global_median=float(np.median(pooled)),
            n_patches=len(per_patch),
        )
        print(dose, out[str(dose)], flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/photon_units.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
