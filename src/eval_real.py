"""Evaluate real-noise denoising on the 20 held-out AI4Life large fields."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import bm3d_anscombe, nlm, tv
from model import UNet
from real_common import load_pair, pairs, split_small

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/real_unet.pt")
ap.add_argument("--n-patch", type=int, default=8)
ap.add_argument("--out", default="results/real_eval.json")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = UNet(32, 1).to(dev); net.load_state_dict(ck["state_dict"]); net.eval()

# tune classical on small validation pairs
_, val, _ = split_small(0)
rng = np.random.default_rng(2026)
items = []
for npath, gpath, _ in val[:20]:
    n, g = load_pair(npath, gpath)
    items.append((n, g))
best = {}
for name, grid, fn in [
        ("bm3d", [0.5, 0.8, 1.0, 1.2, 1.5], lambda n, v: bm3d_anscombe(n, v)),
        ("nlm", [0.5, 0.8, 1.0, 1.5, 2.0], lambda n, v: nlm(n, v)),
        ("tv", [0.01, 0.02, 0.04, 0.08], lambda n, v: tv(n, v))]:
    bv = (-1, None)
    for v in grid:
        vals = [psnr(g, np.clip(fn(n, v), 0, 1), data_range=1) for n, g in items[:10]]
        m = float(np.mean(vals))
        if m > bv[0]: bv = (m, v)
    best[name] = bv[1]
    print("tuned", name, bv, flush=True)

rows = []
for npath, gpath, stem in pairs("large"):
    n, g = load_pair(npath, gpath)
    ps = [(i, j) for i in range(0, n.shape[0] - 256 + 1, 256)
          for j in range(0, n.shape[1] - 256 + 1, 256)]
    step = max(1, len(ps) // a.n_patch)
    ps = ps[::step][:a.n_patch]
    vals = {k: [] for k in ["noisy", "bm3d", "nlm", "tv", "unet"]}
    for i, j in ps:
        x = n[i:i+256, j:j+256]; gt = g[i:i+256, j:j+256]
        with torch.no_grad():
            p = net(torch.as_tensor(x)[None, None].to(dev)).cpu().numpy()[0, 0]
        preds = dict(noisy=x, bm3d=bm3d_anscombe(x, best["bm3d"]),
                     nlm=nlm(x, best["nlm"]), tv=tv(x, best["tv"]), unet=p)
        for k, im in preds.items():
            vals[k].append((psnr(gt, np.clip(im, 0, 1), data_range=1),
                            ssim(gt, np.clip(im, 0, 1), data_range=1)))
    rows.append({k: [float(np.mean([v[0] for v in vs])),
                     float(np.mean([v[1] for v in vs]))] for k, vs in vals.items()})
    print(stem, {k: round(v[0], 2) for k, v in rows[-1].items()}, flush=True)
print("means", {k: round(float(np.mean([r[k][0] for r in rows])), 2)
               for k in rows[0]}, flush=True)
print("ssim", {k: round(float(np.mean([r[k][1] for r in rows])), 3)
              for k in rows[0]}, flush=True)
import json
json.dump(dict(tuned=best, rows=rows), open(a.out, "w"), indent=2)
