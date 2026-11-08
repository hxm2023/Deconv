"""External datasets: TV tuned on 5 fields, U-Net zero-shot on 15 held-out."""
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
LAMS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3]


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
    out = {}
    for name in ["bbbc015", "bbbc016", "bbbc018"]:
        paths, scale = dataset_paths(name)
        sp = split_paths(paths, 0, 5, 15)
        val = [load_image(p, scale) for p in sp["val"]]
        test = [load_image(p, scale) for p in sp["test"]]
        vg = [sample(img) for img in val]
        # TV tuning on the five validation fields
        best = None
        for lam in LAMS:
            scores = []
            for fi in range(len(val)):
                rng = np.random.default_rng(2026)
                ys = [degrade(g, 2.0, 30, rng) for g in vg[fi]]
                scores += [psnr(g, tv_deconv(y, 2.0, lam, rho=0.3, iters=30,
                                             pad=32))
                           for g, y in zip(vg[fi], ys)]
            m = float(np.mean(scores))
            if best is None or m > best[1]:
                best = (lam, m)
        lam = best[0]
        tg = [sample(img) for img in test]
        unet, tv, delta = [], [], []
        for fi in range(len(test)):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, 30, rng) for g in tg[fi]]
            with torch.no_grad():
                p = net(torch.as_tensor(np.asarray(ys))[:, None].to(dev)
                        ).cpu().numpy()[:, 0]
            pu = float(np.mean([psnr(g, q) for g, q in zip(tg[fi], p)]))
            pt = float(np.mean([psnr(g, tv_deconv(y, 2.0, lam, rho=0.3,
                                                  iters=30, pad=32))
                                for g, y in zip(tg[fi], ys)]))
            unet.append(pu); tv.append(pt); delta.append(pu - pt)
        out[name] = dict(lam=lam, val_psnr=best[1], unet=unet, tv=tv,
                         delta=delta, unet_mean=float(np.mean(unet)),
                         tv_mean=float(np.mean(tv)),
                         delta_mean=float(np.mean(delta)),
                         wins=int((np.array(delta) > 0).sum()))
        print(name, "lam", lam, "U", round(out[name]["unet_mean"], 2),
              "TV", round(out[name]["tv_mean"], 2),
              "d", round(out[name]["delta_mean"], 2),
              "wins", out[name]["wins"], flush=True)
    Path("results/external_tuned.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
