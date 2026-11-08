"""PSF-width TV baseline with the tuned rho/iters and reflect padding."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv_deconv
from common import dataset_paths, degrade, load_image, patches, psnr, split_paths


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    out = {}
    for sigma in [1.5, 2.0, 2.5]:
        rows = []
        for img in test:
            gts = sample(img)
            rng = np.random.default_rng(2026)
            ys = [degrade(g, sigma, 30.0, rng) for g in gts]
            p = [tv_deconv(y, sigma, 0.04, rho=0.3, iters=30, pad=32)
                 for y in ys]
            rows.append(float(np.mean([psnr(g, q) for g, q in zip(gts, p)])))
        out[str(sigma)] = rows
        print(sigma, round(float(np.mean(rows)), 2), flush=True)
    Path("results/tvdeconv_sigma_fair.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
