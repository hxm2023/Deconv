"""Evaluate budget-matched per-dose specialists at their own doses."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, ssim, split_paths)
from model import UNet

torch.set_num_threads(2)


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    gts = [sample(img) for img in test]
    out = {}
    for dose in [3, 30, 100]:
        ck = torch.load(f"checkpoints/bbbc014_spec{dose}_strong.pt",
                        map_location=dev, weights_only=False)
        net = UNet(ck.get("args", {}).get("ch", 64), ck["in_ch"]).to(dev)
        net.load_state_dict(ck["state_dict"]); net.eval()
        ps, ss, ff = [], [], []
        for fi in range(len(test)):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in gts[fi]]
            with torch.no_grad():
                p = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                        ).cpu().numpy()[:, 0]
            ps.append(np.mean([psnr(g, q) for g, q in zip(gts[fi], p)]))
            ss.append(np.mean([ssim(g, q) for g, q in zip(gts[fi], p)]))
            ff.append(np.mean([fg_psnr(g, q) for g, q in zip(gts[fi], p)]))
        out[str(dose)] = dict(psnr=float(np.mean(ps)),
                              ssim=float(np.mean(ss)),
                              fg=float(np.mean(ff)))
        print(dose, out[str(dose)], flush=True)
    Path("results/spec_strong.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
