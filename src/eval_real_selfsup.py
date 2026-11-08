"""Evaluate the self-supervised real-noise model on held-out pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import UNet
from real_common import load_pair, split_small

torch.set_num_threads(2)


def checkerboard(h, w, parity):
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    return ((i + j) % 2 == parity).astype(np.float32)


def block_pattern(h, w, block, parity):
    m = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, block):
        for j in range(0, w, block):
            m[i:i + block, j:j + block] = (
                (i // block + j // block) % 2 == parity)
    return m


def make_mask(h, w, block, rng):
    if block <= 0:
        return checkerboard(h, w, int(rng.integers(0, 2)))
    m = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, block):
        for j in range(0, w, block):
            m[i:i + block, j:j + block] = int(rng.integers(0, 2))
    if m.mean() in (0.0, 1.0):
        m = 1.0 - m
    return m


def infer(net, n, block, n_mask, rng):
    out_sum = np.zeros_like(n)
    count = np.zeros_like(n)
    for _ in range(n_mask):
        m = make_mask(n.shape[0], n.shape[1], block, rng)
        with torch.no_grad():
            pred = net(torch.as_tensor(n * m)[None, None]).numpy()[0, 0]
        take = m == 0
        out_sum[take] += pred[take]
        count[take] += 1
    return np.clip(out_sum / np.maximum(count, 1), 0, 1)


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/real_selfsup.pt")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--block", type=int, default=0)
ap.add_argument("--n-mask", type=int, default=2)
ap.add_argument("--ch", type=int, default=32)
ap.add_argument("--out", default="results/real_selfsup_test40.json")
a = ap.parse_args()
dev = "cpu"
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = UNet(a.ch, 1).to(dev); net.load_state_dict(ck["state_dict"]); net.eval()
_, _, test = split_small(0)
if a.n:
    test = test[:a.n]
rng = np.random.default_rng(2026)
rows = []
for npath, gpath, stem in test:
    n, g = load_pair(npath, gpath)
    p = infer(net, n, a.block, a.n_mask, rng)
    rows.append(dict(stem=int(stem),
                     noisy=[float(psnr(g, n, data_range=1)),
                            float(ssim(g, n, data_range=1))],
                     selfsup=[float(psnr(g, p, data_range=1)),
                              float(ssim(g, p, data_range=1))]))
print("n", len(rows), flush=True)
for k in ("noisy", "selfsup"):
    print(k, "PSNR", round(float(np.mean([r[k][0] for r in rows])), 2),
          "SSIM", round(float(np.mean([r[k][1] for r in rows])), 3), flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(rows=rows, ckpt_val=ck.get("val")), open(a.out, "w"), indent=2)
