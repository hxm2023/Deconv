"""Tune TV deconvolution/denoising weights for BBBC039 segmentation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv, tv_deconv
from common import bbbc039_official_split, degrade, save_json
from seg_common import dice, load_pair

DOSES = [1, 3, 10, 30, 100, 300]
LAMS = [0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3, 0.4, 0.6, 0.8]


def otsu(img):
    try:
        return img > threshold_otsu(img)
    except Exception:
        return img > float(img.mean())


def main():
    sp = bbbc039_official_split()
    fovs = sp["val"][::max(1, len(sp["val"]) // 10)][:10]
    items = []
    for path in fovs:
        img, mask = load_pair(path)
        ps = list(__import__("common").patches(img, 128))
        step = max(1, len(ps) // 4)
        for i, j, _ in ps[::step][:4]:
            items.append((img[i:i+128, j:j+128], mask[i:i+128, j:j+128]))
    out = {}
    for P in DOSES:
        rng = np.random.default_rng(2026)
        ys = [degrade(g, 2.0, P, rng) for g, _ in items]
        best_d = (-1, None)
        best_n = (-1, None)
        for w in LAMS:
            d = np.mean([dice(m, otsu(tv_deconv(y, 2, w, iters=30)))
                         for (g, m), y in zip(items, ys)])
            n = np.mean([dice(m, otsu(tv(y, w))) for (g, m), y in zip(items, ys)])
            if d > best_d[0]:
                best_d = (d, w)
            if n > best_n[0]:
                best_n = (n, w)
        out[str(P)] = dict(tvdec_w=best_d[1], tvdec_dice=float(best_d[0]),
                           tvden_w=best_n[1], tvden_dice=float(best_n[0]))
        print(P, out[str(P)], flush=True)
    save_json("results/seg_tv_tuning.json", out)


if __name__ == "__main__":
    main()
