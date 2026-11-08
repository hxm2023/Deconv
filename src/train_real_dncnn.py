"""Supervised DnCNN baseline on the real AI4Life pairs (second deep model)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_common import load_pair, norm99, split_small

ap = argparse.ArgumentParser()
ap.add_argument("--patches", type=int, default=8000)
ap.add_argument("--patch", type=int, default=128)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--ch", type=int, default=64)
ap.add_argument("--out", default="checkpoints/real_dncnn.pt")
a = ap.parse_args()

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
train, val, test = split_small(0)
print("train", len(train), "val", len(val), "device", dev, flush=True)
from skimage import io
cache = []
for npath, gpath, _ in train:
    cache.append((norm99(io.imread(npath).astype(np.float32)),
                  norm99(io.imread(gpath).astype(np.float32))))
print("cached", len(cache), "pairs", flush=True)
rng = np.random.default_rng(0)
X, Y = [], []
for k in range(a.patches):
    n, g = cache[k % len(cache)]
    i = int(rng.integers(0, n.shape[0] - a.patch + 1))
    j = int(rng.integers(0, n.shape[1] - a.patch + 1))
    X.append(n[i:i + a.patch, j:j + a.patch])
    Y.append(g[i:i + a.patch, j:j + a.patch])
X = torch.as_tensor(np.asarray(X), dtype=torch.float32)[:, None]
Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)[:, None]
print("patches", tuple(X.shape), flush=True)


class DnCNN(nn.Module):
    def __init__(self, ch=64, depth=17):
        super().__init__()
        layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(True)]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(ch, ch, 3, padding=1),
                       nn.BatchNorm2d(ch), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)


def eval_val(net):
    net.eval(); vals = []
    with torch.no_grad():
        for npath, gpath, _ in val[:60]:
            n, g = load_pair(npath, gpath)
            p = net(torch.as_tensor(n)[None, None].to(dev)).cpu().numpy()[0, 0]
            mse = float(np.mean((np.clip(p, 0, 1) - g) ** 2))
            vals.append(10 * np.log10(1 / max(mse, 1e-12)))
    net.train(); return float(np.mean(vals))


net = DnCNN(a.ch).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
best = -1; t0 = time.time()
for ep in range(a.epochs):
    idx = torch.randperm(len(X)); tot = 0.0
    for q in range(0, len(X), a.batch):
        mb = idx[q:q + a.batch]
        xb = X[mb].to(dev); yb = Y[mb].to(dev)
        if torch.rand(1).item() < 0.5:
            xb = torch.flip(xb, dims=(3,)); yb = torch.flip(yb, dims=(3,))
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            xb = torch.rot90(xb, k, dims=(2, 3)); yb = torch.rot90(yb, k, dims=(2, 3))
        pred = net(xb); loss = F.mse_loss(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()) * len(mb)
    sched.step(); v = eval_val(net)
    print(f"epoch {ep} loss {tot/len(X):.6f} val {v:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if v > best:
        best = v
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=net.state_dict(), val=v, args=vars(a)), a.out)
        print("  saved", a.out, flush=True)
