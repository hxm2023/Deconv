"""Read-noise robustness for the strong mixed-dose model and fair TV."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv_deconv
from common import (dataset_paths, degrade, load_image, patches, psnr,
                    split_paths)
from model import UNet

torch.set_num_threads(2)
TV = dict(weight=0.04, rho=0.3, iters=30, pad=32)


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_blind_strong.pt", map_location=dev,
                    weights_only=False)
    ch = ck.get("args", {}).get("ch", 32)
    net = UNet(ch, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    gts = [sample(img) for img in test]
    out = {}
    for read in [0.0, 0.005, 0.01]:
        u_all, t_all = [], []
        for fi in range(len(test)):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, 30, rng) for g in gts[fi]]
            if read:
                ys = [np.clip(y + rng.normal(0, read, y.shape), 0, 1)
                      .astype(np.float32) for y in ys]
            with torch.no_grad():
                p = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                        ).cpu().numpy()[:, 0]
            u_all.append(np.mean([psnr(g, q) for g, q in zip(gts[fi], p)]))
            t_all.append(np.mean([psnr(g, tv_deconv(y, 2.0, **TV))
                                  for g, y in zip(gts[fi], ys)]))
        out[str(read)] = dict(unet=float(np.mean(u_all)),
                              tvdec=float(np.mean(t_all)))
        print(read, round(out[str(read)]["unet"], 2),
              round(out[str(read)]["tvdec"], 2), flush=True)
    Path("results/read_noise_fair.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
