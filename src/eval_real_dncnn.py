"""Evaluate the supervised DnCNN baseline on held-out real pairs."""
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
from real_common import load_pair, split_small

torch.set_num_threads(2)


class DnCNN(torch.nn.Module):
    def __init__(self, ch=64, depth=17):
        super().__init__()
        layers = [torch.nn.Conv2d(1, ch, 3, padding=1), torch.nn.ReLU(True)]
        for _ in range(depth - 1):
            layers += [torch.nn.Conv2d(ch, ch, 3, padding=1),
                       torch.nn.BatchNorm2d(ch), torch.nn.ReLU(True)]
        layers += [torch.nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/real_dncnn.pt")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", default="results/real_dncnn_test40.json")
a = ap.parse_args()
dev = "cpu"
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = DnCNN(ck["args"].get("ch", 64)).to(dev)
net.load_state_dict(ck["state_dict"]); net.eval()
_, _, test = split_small(0)
if a.n:
    test = test[:a.n]
rows = []
for npath, gpath, stem in test:
    n, g = load_pair(npath, gpath)
    with torch.no_grad():
        p = net(torch.as_tensor(n)[None, None]).numpy()[0, 0]
    rows.append(dict(stem=int(stem),
                     dncnn=[float(psnr(g, np.clip(p, 0, 1), data_range=1)),
                            float(ssim(g, np.clip(p, 0, 1), data_range=1))]))
print("n", len(rows), "dncnn PSNR",
      round(float(np.mean([r["dncnn"][0] for r in rows])), 2),
      "SSIM", round(float(np.mean([r["dncnn"][1] for r in rows])), 3),
      flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(rows=rows, ckpt_val=ck.get("val")), open(a.out, "w"), indent=2)
