"""Pre-experiment: headroom for blind, spatially varying PSF deconvolution.

Each test patch is blurred with its own PSF width. A classical method with a
nominal PSF is compared with an oracle that knows the per-patch width. A large
gap means a model that infers the local PSF from the image could add a real
capability that non-AI deconvolution cannot provide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv_deconv
from common import dataset_paths, degrade, load_image, patches, psnr, split_paths
from model import UNet


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_blind3.pt", map_location=dev,
                    weights_only=False)
    net = UNet(32, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    rng = np.random.default_rng(2026)
    rows = {"nominal": [], "oracle": [], "unet": [], "sigma": []}
    for img in test:
        gts = sample(img)
        for g in gts:
            sigma = float(rng.uniform(1.5, 3.0))
            y = degrade(g, sigma, 30.0, rng)
            p_nom = tv_deconv(y, 2.0, 0.04, rho=0.3, iters=30, pad=32)
            p_or = tv_deconv(y, sigma, 0.04, rho=0.3, iters=30, pad=32)
            with torch.no_grad():
                p_un = net(torch.as_tensor(y)[None, None].to(dev)).cpu().numpy()[0, 0]
            rows["nominal"].append(psnr(g, p_nom))
            rows["oracle"].append(psnr(g, p_or))
            rows["unet"].append(psnr(g, p_un))
            rows["sigma"].append(sigma)
    out = {k: float(np.mean(v)) for k, v in rows.items() if k != "sigma"}
    out["headroom_tv"] = out["oracle"] - out["nominal"]
    out["headroom_unet"] = out["oracle"] - out["unet"]
    out["n"] = len(rows["nominal"])
    out["sigma_range"] = [float(min(rows["sigma"])), float(max(rows["sigma"]))]
    print(out, flush=True)
    Path("results/psf_headroom.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
