"""TV-regularized deconvolution on all 80 test patches per FOV."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv_deconv
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, save_json, split_paths, ssim)

LAM = {3: 0.2, 10: 0.08, 30: 0.04, 100: 0.04, 300: 0.04}


def main():
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    out = {}
    for dose in [3, 10, 30, 100, 300]:
        t0 = time.time()
        rows = []
        for img in test:
            ps = list(patches(img, 128))
            rng = np.random.default_rng(2026)
            vals = []
            for _, _, gt in ps:
                y = degrade(gt, 2.0, dose, rng)
                p = tv_deconv(y, 2.0, LAM[dose], iters=30)
                vals.append((psnr(gt, p), ssim(gt, p), fg_psnr(gt, p)))
            a = np.asarray(vals)
            rows.append(dict(psnr=float(a[:, 0].mean()),
                             ssim=float(a[:, 1].mean()),
                             fg=float(a[:, 2].mean())))
        out[str(dose)] = rows
        m = np.array([[r["psnr"], r["ssim"], r["fg"]] for r in rows])
        print(f"dose {dose}: TVdec PSNR {m[:,0].mean():.2f} "
              f"SSIM {m[:,1].mean():.4f} FG {m[:,2].mean():.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    save_json("results/tvdeconv_full.json", out)


if __name__ == "__main__":
    main()
