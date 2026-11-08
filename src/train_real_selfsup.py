"""Self-supervised (J-invariant) real-noise denoising on AI4Life noisy images.

Training uses noisy images only; the clean references are used only for the
final held-out evaluation. The gate: beat tuned BM3D (20.92 dB on the 40
held-out pairs) without any clean training target.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import UNet
from real_common import load_pair, norm99, split_small

ap = argparse.ArgumentParser()
ap.add_argument("--patches", type=int, default=8000)
ap.add_argument("--patch", type=int, default=128)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--block", type=int, default=0,
                help="0 = checkerboard; >0 = random blocks of this size")
ap.add_argument("--ch", type=int, default=32)
ap.add_argument("--out", default="checkpoints/real_selfsup.pt")
a = ap.parse_args()

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
train, val, test = split_small(0)
print("train", len(train), "val", len(val), "device", dev, flush=True)


def checkerboard(h, w, parity=0):
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    return ((i + j) % 2 == parity).astype(np.float32)


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


rng = np.random.default_rng(0)
X, T, M = [], [], []
for k in range(a.patches):
    npath, _, _ = train[k % len(train)]
    n = norm99(__import__("skimage").io.imread(npath).astype(np.float32))
    i = int(rng.integers(0, n.shape[0] - a.patch + 1))
    j = int(rng.integers(0, n.shape[1] - a.patch + 1))
    p = n[i:i + a.patch, j:j + a.patch]
    m = make_mask(a.patch, a.patch, a.block, rng)
    X.append(p * m)
    T.append(p)
    M.append(m)
X = torch.as_tensor(np.asarray(X), dtype=torch.float32)[:, None]
T = torch.as_tensor(np.asarray(T), dtype=torch.float32)[:, None]
M = torch.as_tensor(np.asarray(M), dtype=torch.float32)[:, None]
print("patches", tuple(X.shape), flush=True)


def masked_loss(pred, target, mask):
    keep = 1.0 - mask
    return ((pred - target) ** 2 * keep).sum() / keep.sum().clamp_min(1.0)


def eval_val(net):
    net.eval(); losses = []
    with torch.no_grad():
        for npath, _, _ in val[:60]:
            n = norm99(__import__("skimage").io.imread(npath).astype(np.float32))
            m = torch.as_tensor(make_mask(n.shape[0], n.shape[1], a.block,
                                          np.random.default_rng(1)))[None, None]
            x = torch.as_tensor(n)[None, None].to(dev) * m.to(dev)
            t = torch.as_tensor(n)[None, None].to(dev)
            losses.append(float(masked_loss(net(x), t, m.to(dev))))
    net.train(); return float(np.mean(losses))


net = UNet(a.ch, 1).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
best = 1e9; t0 = time.time()
for ep in range(a.epochs):
    idx = torch.randperm(len(X)); tot = 0.0
    for q in range(0, len(X), a.batch):
        mb = idx[q:q + a.batch]
        xb = X[mb].to(dev); yb = T[mb].to(dev); mb_m = M[mb].to(dev)
        pred = net(xb)
        loss = masked_loss(pred, yb, mb_m)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()) * len(mb)
    sched.step(); v = eval_val(net)
    print(f"epoch {ep} loss {tot/len(X):.6f} val {v:.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if v < best:
        best = v
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=net.state_dict(), val=v, args=vars(a)), a.out)
        print("  saved", a.out, flush=True)
