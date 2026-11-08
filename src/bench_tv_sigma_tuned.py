"""Tune TV lambda per PSF width and evaluate the 30-photon test FOVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv_deconv
from common import dataset_paths, degrade, load_image, patches, psnr, split_paths

LAMS = [0.01, 0.02, 0.04, 0.08, 0.12, 0.2]


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    val = [load_image(p, scale) for p in sp["val"]][:5]
    test = [load_image(p, scale) for p in sp["test"]]
    vg = [g for img in val for g in sample(img)]
    out = {}
    params = {}
    for sigma in [1.5, 2.0, 2.5]:
        rng = np.random.default_rng(2026)
        vy = [degrade(g, sigma, 30.0, rng) for g in vg]
        scores = {}
        for lam in LAMS:
            scores[lam] = float(np.mean([
                psnr(g, tv_deconv(y, sigma, lam, rho=0.3, iters=30, pad=32))
                for g, y in zip(vg, vy)]))
        lam = max(LAMS, key=scores.get)
        params[str(sigma)] = dict(weight=lam, val=scores)
        rows = []
        for img in test:
            gts = sample(img)
            rng = np.random.default_rng(2026)
            ys = [degrade(g, sigma, 30.0, rng) for g in gts]
            p = [tv_deconv(y, sigma, lam, rho=0.3, iters=30, pad=32)
                 for y in ys]
            rows.append(float(np.mean([psnr(g, q) for g, q in zip(gts, p)])))
        out[str(sigma)] = rows
        print(sigma, "lam", lam, "test", round(float(np.mean(rows)), 2),
              flush=True)
    Path("results/tvdeconv_sigma_tuned.json").write_text(
        json.dumps(dict(rows=out, params=params), indent=2))


if __name__ == "__main__":
    main()
