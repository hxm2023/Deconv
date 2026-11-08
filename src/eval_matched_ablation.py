"""Evaluate specialists, blind, scrambled, and conditioned models on identical patches."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, save_json, split_paths, ssim)
from model import UNet


def predict(net, ys, dose, dev, dose_cond, patch=128, batch=64):
    import math
    c = (math.log10(dose) - math.log10(3.0)) / (math.log10(300.0) - math.log10(3.0))
    out = []
    with torch.no_grad():
        for q in range(0, len(ys), batch):
            x = torch.as_tensor(np.asarray(ys[q:q + batch]),
                                dtype=torch.float32)[:, None]
            if dose_cond:
                ch = torch.full((len(x), 1, patch, patch), c)
                x = torch.cat([x, ch], 1)
            out.append(net(x.to(dev)).cpu().numpy()[:, 0])
    return np.concatenate(out, 0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    gts = []
    for img in test:
        ps = list(patches(img, 128))
        step = max(1, len(ps) // 10)
        gts.append([g for _, _, g in ps[::step][:10]])
    models = {
        "dcond": ("checkpoints/bbbc014_dcond.pt", [3, 30, 100]),
        "specialist": (None, [3, 30, 100]),
        "blind": ("checkpoints/bbbc014_blind3.pt", [3, 30, 100]),
        "scr": ("checkpoints/bbbc014_scr2.pt", [3, 30, 100]),
    }
    specs = {3: "checkpoints/bbbc014_spec3.pt",
             30: "checkpoints/bbbc014_spec30b.pt",
             100: "checkpoints/bbbc014_spec100.pt"}
    out = {}
    for name, (path, doses) in models.items():
        out[name] = {}
        for dose in doses:
            p = specs[dose] if name == "specialist" else path
            ck = torch.load(p, map_location=dev, weights_only=False)
            net = UNet(32, ck["in_ch"]).to(dev)
            net.load_state_dict(ck["state_dict"]); net.eval()
            vals = []
            for fi, img in enumerate(test):
                rng = np.random.default_rng(2026)
                ys = [degrade(g, 2.0, dose, rng) for g in gts[fi]]
                pred = predict(net, ys, dose, dev, ck["dose_cond"])
                for g, pr in zip(gts[fi], pred):
                    vals.append((psnr(g, pr), ssim(g, pr), fg_psnr(g, pr)))
            a = np.asarray(vals)
            out[name][str(dose)] = dict(
                psnr=float(a[:, 0].mean()), ssim=float(a[:, 1].mean()),
                fg=float(a[:, 2].mean()))
            print(name, dose, out[name][str(dose)], flush=True)
    save_json("results/matched_ablation.json", out)


if __name__ == "__main__":
    main()
