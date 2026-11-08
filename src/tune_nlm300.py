"""Tune NLM at P=300 on validation and evaluate the test FOVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import nlm
from common import dataset_paths, degrade, load_image, patches, psnr, split_paths

H = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]


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
    rng = np.random.default_rng(2026)
    vy = [degrade(g, 2.0, 300.0, rng) for g in vg]
    scores = {}
    for h in H:
        scores[h] = float(np.mean([psnr(g, nlm(y, h))
                                   for g, y in zip(vg, vy)]))
    h = max(H, key=scores.get)
    rows = []
    for img in test:
        gts = sample(img)
        rng = np.random.default_rng(2026)
        ys = [degrade(g, 2.0, 300.0, rng) for g in gts]
        rows.append(float(np.mean([psnr(g, nlm(y, h))
                                   for g, y in zip(gts, ys)])))
    print("h", h, "scores", scores, "test", round(float(np.mean(rows)), 2),
          flush=True)
    Path("results/nlm300.json").write_text(json.dumps(
        dict(h=h, scores=scores, rows=rows), indent=2))


if __name__ == "__main__":
    main()
