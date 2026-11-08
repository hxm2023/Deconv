"""Evaluate a dose-conditioned U-Net across photon doses on held-out FOVs."""
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
from common import (bbbc039_official_split, dataset_paths, degrade, fg_psnr,
                    load_image, patches, psnr, save_json, split_paths, ssim)
from model import UNet


def cond(photons):
    lo, hi = math.log10(3.0), math.log10(300.0)
    return (math.log10(photons) - lo) / (hi - lo)


def predict(net, ys, photons, dev, dose_cond, batch=64):
    outs = []
    with torch.no_grad():
        for q in range(0, len(ys), batch):
            xb = torch.as_tensor(np.asarray(ys[q:q + batch]),
                                 dtype=torch.float32)[:, None]
            if dose_cond:
                h, w = xb.shape[-2:]
                c = torch.full((len(xb), 1, h, w), cond(photons),
                               dtype=torch.float32)
                xb = torch.cat([xb, c], 1)
            outs.append(net(xb.to(dev)).cpu().numpy()[:, 0])
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bbbc014_dcond.pt")
    ap.add_argument("--dataset", default="bbbc014")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--doses", default="3,5,10,20,30,50,100,200,300")
    ap.add_argument("--seeds", default="31,32,33")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--out", default="results/unet_dose_response.json")
    a = ap.parse_args()
    doses = [float(x) for x in a.doses.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = UNet(32, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    print("loaded", a.ckpt, "val", ck.get("val_psnr"), "device", dev, flush=True)

    paths, scale = dataset_paths(a.dataset, a.channel)
    if a.dataset == "bbbc039":
        sp = bbbc039_official_split()
    else:
        sp = split_paths(paths, a.split_seed, a.n_val, a.n_test)
    test = [load_image(p, scale) for p in sp["test"]]

    rows = {d: [] for d in doses}
    for d in doses:
        recs = []
        for img in test:
            vals = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                ps = list(patches(img, a.patch))
                gts = [gt for _, _, gt in ps]
                ys = [degrade(gt, 2.0, d, rng) for gt in gts]
                pred = predict(net, ys, d, dev, ck["dose_cond"])
                for gt, p in zip(gts, pred):
                    vals.append((psnr(gt, p), ssim(gt, p), fg_psnr(gt, p)))
            a_ = np.array(vals)
            recs.append(dict(fov_mean_psnr=float(a_[:, 0].mean()),
                             fov_mean_ssim=float(a_[:, 1].mean()),
                             fov_mean_fg=float(a_[:, 2].mean())))
        rows[d] = recs
        m = np.array([[r["fov_mean_psnr"], r["fov_mean_ssim"],
                       r["fov_mean_fg"]] for r in recs])
        print(f"dose {d:5.0f}: PSNR {m[:,0].mean():.2f} "
              f"(p10 {np.percentile(m[:,0],10):.2f}) SSIM {m[:,1].mean():.4f} "
              f"FG {m[:,2].mean():.2f}", flush=True)

    save_json(a.out, dict(ckpt=a.ckpt, dataset=a.dataset, doses=doses,
                          seeds=seeds, test=rows))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
