"""Evaluate the budget-matched phase-scrambled control at 3/30/100."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (dataset_paths, degrade, load_image, patches, psnr,
                    split_paths)
from model import UNet

torch.set_num_threads(2)


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_scr_blind_strong.pt",
                    map_location=dev, weights_only=False)
    ch = ck.get("args", {}).get("ch", 64)
    net = UNet(ch, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    gts = [sample(img) for img in test]
    out = {}
    for dose in [3, 30, 100]:
        rows = []
        for fi in range(len(test)):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in gts[fi]]
            with torch.no_grad():
                p = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                        ).cpu().numpy()[:, 0]
            rows.append(float(np.mean([psnr(g, q)
                                       for g, q in zip(gts[fi], p)])))
        out[str(dose)] = rows
        print(dose, round(float(np.mean(rows)), 2), flush=True)
    Path("results/scr_blind_strong.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
