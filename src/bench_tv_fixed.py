"""Evaluate the tuned TV-ADMM test rows with the parameters selected on val."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv, tv_deconv
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, ssim, split_paths)

DOSES = [3, 10, 30, 100, 300]
PARAMS = {3: (0.20, 0.3, 30), 10: (0.08, 0.3, 30), 30: (0.04, 0.3, 30),
          100: (0.02, 3.0, 30), 300: (0.01, 3.0, 30)}
DN_W = {3: 0.4, 10: 0.3, 30: 0.12, 100: 0.04, 300: 0.02}


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
        lam, rho, it = PARAMS[dose]
        rows = []
        for img in test:
            gts = sample(img)
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in gts]
            p_tv = [tv_deconv(y, 2.0, lam, rho=rho, iters=it, pad=32)
                    for y in ys]
            p_dn = [tv(y, DN_W[dose]) for y in ys]
            rows.append(dict(
                tv_deconv=[psnr(g, p) for g, p in zip(gts, p_tv)],
                tv_denoise=[psnr(g, p) for g, p in zip(gts, p_dn)],
                tv_deconv_ssim=[ssim(g, p) for g, p in zip(gts, p_tv)],
                tv_denoise_ssim=[ssim(g, p) for g, p in zip(gts, p_dn)],
                tv_deconv_fg=[fg_psnr(g, p) for g, p in zip(gts, p_tv)],
                tv_denoise_fg=[fg_psnr(g, p) for g, p in zip(gts, p_dn)],
            ))
        out[str(dose)] = dict(lam=lam, rho=rho, iters=it, rows=rows)
        print(dose, "TVdec", round(float(np.mean([np.mean(r["tv_deconv"])
                                                  for r in rows])), 2),
              "TVden", round(float(np.mean([np.mean(r["tv_denoise"])
                                            for r in rows])), 2), flush=True)
    Path("results/tvdeconv_fixed.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
