"""Strong model at doses not used during training (5, 50, 200)."""
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
    ck = torch.load("checkpoints/bbbc014_blind_strong.pt", map_location=dev,
                    weights_only=False)
    net = UNet(ck.get("args", {}).get("ch", 64), ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    gts = [sample(img) for img in test]
    out = {}
    for dose in [5, 50, 200]:
        rows = []
        for fi in range(len(test)):
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
        print(dose, round(float(np.mean([r["psnr"] for r in rows])), 2),
              round(float(np.mean([r["ssim"] for r in rows])), 3),
              flush=True)

    B = json.load(open("results/blind_fair_doses.json"))
    tr = {d: float(np.mean([r["psnr"] for r in B[str(d)]]))
          for d in [3, 10, 30, 100, 300]}
    checks = []
    for lo, hi, mid in [(3, 10, 5), (30, 100, 50), (100, 300, 200)]:
        w = ((np.log10(mid) - np.log10(lo)) /
             (np.log10(hi) - np.log10(lo)))
        interp = tr[lo] * (1 - w) + tr[hi] * w
        got = float(np.mean([r["psnr"] for r in out[str(mid)]]))
        checks.append(dict(dose=mid, lo=lo, hi=hi, interp=interp, measured=got,
                           residual=got - interp))
        print("dose", mid, "interp", round(interp, 2), "measured",
              round(got, 3), "residual", round(got - interp, 3), flush=True)
    Path("results/heldout_doses.json").write_text(json.dumps(
        dict(rows=out, interpolation=checks), indent=2))


if __name__ == "__main__":
    main()
