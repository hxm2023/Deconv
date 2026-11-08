"""Evaluate the P=1 extended checkpoint with fixed tuned TV parameters."""
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
LAM, RHO, IT = 0.3, 1.0, 30


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_blind_p1_strong.pt",
                    map_location=dev, weights_only=False)
    net = UNet(ck.get("args", {}).get("ch", 64), ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    rows = []
    for fi in range(len(test)):
        gts = sample(test[fi])
        rng = np.random.default_rng(2026)
        ys = [degrade(g, 2.0, 1, rng) for g in gts]
        with torch.no_grad():
            p = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                    ).cpu().numpy()[:, 0]
        p_tv = [tv_deconv(y, 2.0, LAM, rho=RHO, iters=IT, pad=32)
                for y in ys]
        rows.append(dict(
            unet=float(np.mean([psnr(g, q) for g, q in zip(gts, p)])),
            tv=float(np.mean([psnr(g, q) for g, q in zip(gts, p_tv)]))))
    u = np.array([r["unet"] for r in rows])
    t = np.array([r["tv"] for r in rows])
    out = dict(lam=LAM, rho=RHO, iters=IT, rows=rows,
               unet_mean=float(u.mean()), tv_mean=float(t.mean()),
               gain=float((u - t).mean()), wins=int((u > t).sum()))
    print("P1", round(u.mean(), 2), round(t.mean(), 2),
          round(float((u - t).mean()), 2), int((u > t).sum()), flush=True)
    Path("results/p1_fixed.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
