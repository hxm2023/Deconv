"""Fair real-data check on held-out small AI4Life pairs (field-clean split)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import bm3d_anscombe, nlm, tv
from model import UNet
from real_common import load_pair, split_small

torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/real_unet.pt")
ap.add_argument("--n", type=int, default=0, help="0 = all test pairs")
ap.add_argument("--out", default="results/real_small_test.json")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = UNet(32, 1).to(dev)
net.load_state_dict(ck["state_dict"])
net.eval()

_, val, test = split_small(0)
if a.n:
    test = test[:a.n]
print("test pairs", len(test), "device", dev, flush=True)

# classical parameters tuned on small validation pairs (never test)
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
        m = float(np.mean([psnr(g, np.clip(fn(n, v), 0, 1), data_range=1)
                           for n, g in items[:10]]))
        if m > bv[0]:
            bv = (m, v)
    best[name] = bv[1]
    print("tuned", name, bv, flush=True)

rows = []
t0 = time.time()
for k, (npath, gpath, stem) in enumerate(test):
    n, g = load_pair(npath, gpath)
    with torch.no_grad():
        p = net(torch.as_tensor(n)[None, None].to(dev)).cpu().numpy()[0, 0]
    preds = dict(noisy=n, bm3d=bm3d_anscombe(n, best["bm3d"]),
                 nlm=nlm(n, best["nlm"]), tv=tv(n, best["tv"]), unet=p)
    rows.append({kk: [float(psnr(g, np.clip(im, 0, 1), data_range=1)),
                      float(ssim(g, np.clip(im, 0, 1), data_range=1))]
                 for kk, im in preds.items()})
    if (k + 1) % 20 == 0:
        print(k + 1, "of", len(test), f"{time.time()-t0:.1f}s", flush=True)

print("elapsed", round(time.time() - t0, 1), flush=True)
summary = {}
for kk in rows[0]:
    summary[kk] = [float(np.mean([r[kk][0] for r in rows])),
                   float(np.mean([r[kk][1] for r in rows]))]
    print(kk, "PSNR", round(summary[kk][0], 2), "SSIM",
          round(summary[kk][1], 3), flush=True)

from scipy.stats import wilcoxon
for kk in ["bm3d", "nlm", "tv"]:
    d = [r["unet"][0] - r[kk][0] for r in rows]
    w = wilcoxon(d)
    print(f"unet-{kk}: mean {np.mean(d):+.2f} dB, wins {sum(x>0 for x in d)}/{len(d)}, "
          f"p={w.pvalue:.2e}", flush=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(tuned=best, rows=rows, summary=summary), open(a.out, "w"), indent=2)
