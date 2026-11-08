"""Evaluate the larger mixed-dose U-Net on the same test benchmark."""
from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bbbc014_blind_strong.pt")
    ap.add_argument("--out", default="results/blind_strong_doses.json")
    aa = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(aa.ckpt, map_location=dev,
                    weights_only=False)
    ch = ck.get("args", {}).get("ch", 32)
    net = UNet(ch, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    gts = [sample(img) for img in test]
    out = {}
    tv = json.load(open("results/tvdeconv_doses.json"))
    for dose in [3, 10, 30, 100, 300]:
        rows = []
        for fi, img in enumerate(test):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in gts[fi]]
            with torch.no_grad():
                p = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                        ).cpu().numpy()[:, 0]
            rows.append(dict(
                psnr=float(np.mean([psnr(g, q) for g, q in zip(gts[fi], p)])),
                ssim=float(np.mean([ssim(g, q) for g, q in zip(gts[fi], p)])),
                fg=float(np.mean([fg_psnr(g, q) for g, q in zip(gts[fi], p)]))))
        out[str(dose)] = rows
        t = np.array([np.mean(r["tv_deconv"]) for r in tv[str(dose)]["rows"]])
        u = np.array([r["psnr"] for r in rows])
        print(dose, "unet", round(float(u.mean()), 2), "tv", round(float(t.mean()), 2),
              "gain", round(float((u - t).mean()), 2), flush=True)
    Path(aa.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
