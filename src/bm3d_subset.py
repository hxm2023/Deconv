"""BM3D comparison on a 100-patch subset at 3 and 30 photons."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import bm3d_anscombe, tv
from common import (dataset_paths, degrade, load_image, patches, psnr, save_json,
                    split_paths, ssim)
from model import UNet


def unet_predict(net, y, photons, dev):
    import math
    lo, hi = math.log10(3.0), math.log10(300.0)
    c = (math.log10(photons) - lo) / (hi - lo)
    x = torch.as_tensor(y, dtype=torch.float32)[None, None].to(dev)
    ch = torch.full((1, 1, y.shape[0], y.shape[1]), c, device=dev)
    with torch.no_grad():
        return net(torch.cat([x, ch], 1)).cpu().numpy()[0, 0]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_dcond.pt", map_location=dev,
                    weights_only=False)
    net = UNet(32, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    # tune BM3D sigma on validation patches
    val = [load_image(p, scale) for p in sp["val"]]
    best = (-1, None)
    for sig in [0.5, 0.8, 1.0, 1.2, 1.5]:
        vals = []
        rng = np.random.default_rng(9)
        for img in val[:10]:
            gt = list(patches(img, 128))[0][2]
            y = degrade(gt, 2.0, 3.0, rng)
            vals.append(psnr(gt, bm3d_anscombe(y, sig)))
        m = float(np.mean(vals))
        if m > best[0]:
            best = (m, sig)
    sig = best[1]
    print("tuned BM3D sigma_psd", sig, "val", round(best[0], 2), flush=True)
    out = {}
    for dose in [3.0, 30.0]:
        rows = {m: [] for m in ["bm3d", "tv", "unet"]}
        for img in test:
            ps = list(patches(img, 128))
            step = max(1, len(ps) // 5)
            for _, _, gt in ps[::step][:5]:
                rng = np.random.default_rng(2026)
                y = degrade(gt, 2.0, dose, rng)
                for m, p in [("bm3d", bm3d_anscombe(y, sig)),
                             ("tv", tv(y, 0.4 if dose == 3 else 0.12)),
                             ("unet", unet_predict(net, y, dose, dev))]:
                    rows[m].append((psnr(gt, p), ssim(gt, p)))
        out[str(dose)] = {m: dict(psnr=float(np.mean([x[0] for x in v])),
                                  ssim=float(np.mean([x[1] for x in v])))
                          for m, v in rows.items()}
        print(dose, out[str(dose)], flush=True)
    save_json("results/bm3d_subset.json", dict(sigma_psd=sig, rows=out))


if __name__ == "__main__":
    main()
