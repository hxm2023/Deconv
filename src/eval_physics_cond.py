"""Evaluate a dose- and/or PSF-conditioned U-Net across photon dose and PSF."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, save_json, split_paths, ssim)
from model import UNet


def cond_dose(photons):
    lo, hi = math.log10(3.0), math.log10(300.0)
    return (math.log10(photons) - lo) / (hi - lo)


def cond_sigma(sigma):
    return (sigma - 1.5) / 1.0


def predict(net, ys, photons, sigma, dev, dose_cond, sigma_cond, batch=64):
    outs = []
    with torch.no_grad():
        for q in range(0, len(ys), batch):
            xb = torch.as_tensor(np.asarray(ys[q:q + batch]),
                                 dtype=torch.float32)[:, None]
            h, w = xb.shape[-2:]
            ch = []
            if dose_cond:
                ch.append(torch.full((len(xb), 1, h, w), cond_dose(photons)))
            if sigma_cond:
                ch.append(torch.full((len(xb), 1, h, w), cond_sigma(sigma)))
            if ch:
                xb = torch.cat([xb] + ch, 1)
            outs.append(net(xb.to(dev)).cpu().numpy()[:, 0])
    return np.concatenate(outs, 0)


def sample_patches(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="bbbc014")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--doses", default="3,10,30,100")
    ap.add_argument("--sigmas", default="1.5,2.0,2.5")
    ap.add_argument("--seeds", default="51,52")
    ap.add_argument("--out", default="results/physics_cond.json")
    a = ap.parse_args()
    doses = [float(x) for x in a.doses.split(",")]
    sigmas = [float(x) for x in a.sigmas.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = UNet(32, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    print("loaded", a.ckpt, "dose_cond", ck["dose_cond"], "sigma_cond",
          ck["sigma_cond"], flush=True)

    paths, scale = dataset_paths(a.dataset, a.channel)
    sp = split_paths(paths, a.split_seed, a.n_val, a.n_test)
    test = [load_image(p, scale) for p in sp["test"]]

    rows = {}
    for sigma in sigmas:
        rows[str(sigma)] = {}
        for dose in doses:
            recs = []
            for img in test:
                gts = sample_patches(img)
                vals = []
                for sd in seeds:
                    rng = np.random.default_rng(sd)
                    ys = [degrade(gt, sigma, dose, rng) for gt in gts]
                    pred = predict(net, ys, dose, sigma, dev,
                                   ck["dose_cond"], ck["sigma_cond"])
                    for gt, p in zip(gts, pred):
                        vals.append((psnr(gt, p), ssim(gt, p), fg_psnr(gt, p)))
                ar = np.array(vals)
                recs.append(dict(psnr=float(ar[:, 0].mean()),
                                 ssim=float(ar[:, 1].mean()),
                                 fg=float(ar[:, 2].mean())))
            rows[str(sigma)][str(dose)] = recs
            m = np.array([[r["psnr"], r["ssim"], r["fg"]] for r in recs])
            print(f"sigma {sigma:g} dose {dose:5.0f}: "
                  f"PSNR {m[:,0].mean():.2f} SSIM {m[:,1].mean():.4f} "
                  f"FG {m[:,2].mean():.2f}", flush=True)
    save_json(a.out, dict(ckpt=a.ckpt, doses=doses, sigmas=sigmas,
                          seeds=seeds, rows=rows))


if __name__ == "__main__":
    main()
