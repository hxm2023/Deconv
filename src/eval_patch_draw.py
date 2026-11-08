"""Patch-draw robustness: repeat U-Net/TV comparison over random patch draws."""
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
DRAWS = [2026, 7, 11, 23, 42]
NP = [4, 10]
TV = {3: (0.20, 0.3, 30), 30: (0.04, 0.3, 30), 300: (0.01, 3.0, 30)}


def grid_patches(img, size=128):
    return [(i, j) for i, j, _ in patches(img, size)]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_blind_strong.pt", map_location=dev,
                    weights_only=False)
    net = UNet(ck.get("args", {}).get("ch", 64), ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    coords = [grid_patches(img) for img in test]
    out = {}
    for seed in DRAWS:
        for n in NP:
            key = f"{seed}_{n}"
            rows = []
            for d in [3, 30, 300]:
                lam, rho, it = TV[d]
                u_all, t_all = [], []
                for fi in range(len(test)):
                    rng = np.random.default_rng(seed + fi)
                    idx = rng.choice(len(coords[fi]), n, replace=False)
                    gts, ys = [], []
                    for k in idx:
                        i, j = coords[fi][k]
                        gt = test[fi][i:i + 128, j:j + 128]
                        gts.append(gt)
                        ys.append(degrade(gt, 2.0, d, rng))
                    with torch.no_grad():
                        p = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                                ).cpu().numpy()[:, 0]
                    p_tv = [tv_deconv(y, 2.0, lam, rho=rho, iters=it, pad=32)
                            for y in ys]
                    u_all.append(np.mean([psnr(g, q)
                                          for g, q in zip(gts, p)]))
                    t_all.append(np.mean([psnr(g, q)
                                          for g, q in zip(gts, p_tv)]))
                rows.append(dict(dose=d, unet=float(np.mean(u_all)),
                                 tv=float(np.mean(t_all)),
                                 gain=float(np.mean(np.array(u_all) -
                                                    np.array(t_all)))))
            out[key] = rows
            print(key, [(r["dose"], round(r["gain"], 2)) for r in rows],
                  flush=True)
    Path("results/patch_draw.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
