"""Clean FOV-level benchmark: U-Net vs tuned classical deconvolution.

Baselines are tuned on validation FOVs only. Test FOVs are never used for
tuning or model selection.
"""
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
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import HAS_BM3D, bm3d_anscombe, nlm, rl, tv, wiener_deconv
from common import (bbbc039_official_split, dataset_paths, degrade, fg_psnr,
                    load_image, patches, psnr, save_json, split_paths, ssim)
from model import UNet

GRIDS = dict(tv=[0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4],
             wiener=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
             rl=[5, 10, 20, 30, 50, 100],
             nlm=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])


def tune(fovs, sigma, photons, seed=7):
    rng = np.random.default_rng(seed)
    items = []
    for img in fovs:
        ps = list(patches(img, 128))
        step = max(1, len(ps) // 5)
        for _, _, gt in ps[::step][:5]:
            items.append((gt, degrade(gt, sigma, photons, rng)))
    scores = {}
    params = {}
    for name, grid in GRIDS.items():
        if name == "bm3d" and not HAS_BM3D:
            continue
        print(f"  tuning {name} over {len(grid)} values", flush=True)
        best = (-1.0, None)
        for v in grid:
            vals = []
            for gt, y in items:
                if name == "tv":
                    p = tv(y, v)
                elif name == "wiener":
                    p = wiener_deconv(y, sigma, v)
                elif name == "rl":
                    p = rl(y, sigma, v)
                elif name == "nlm":
                    p = nlm(y, v)
                else:
                    p = bm3d_anscombe(y, v)
                vals.append(psnr(gt, p))
            m = float(np.mean(vals))
            if m > best[0]:
                best = (m, v)
        params[name] = best[1]
        scores[name] = best[0]
    return params, scores


def apply_baselines(y, params, sigma):
    out = dict(tv=tv(y, params["tv"]),
               wiener=wiener_deconv(y, sigma, params["wiener"]),
               rl=rl(y, sigma, params["rl"]),
               nlm=nlm(y, params["nlm"]))
    if "bm3d" in params:
        out["bm3d"] = bm3d_anscombe(y, params["bm3d"])
    return out


def unet_predict(net, ys, photons, sigma, dev, dose_cond, sigma_cond, batch=64):
    outs = []
    net.eval()
    with torch.no_grad():
        for q in range(0, len(ys), batch):
            xb = torch.as_tensor(np.asarray(ys[q:q + batch]),
                                 dtype=torch.float32)[:, None]
            cs = []
            if dose_cond:
                lo, hi = math.log10(3.0), math.log10(300.0)
                cs.append((math.log10(photons) - lo) / (hi - lo))
            if sigma_cond:
                cs.append((sigma - 1.5) / 1.0)
            if cs:
                h, w = xb.shape[-2:]
                c = torch.as_tensor(np.asarray(cs, dtype=np.float32))[None, :, None, None]
                c = c.expand(len(xb), -1, h, w).contiguous()
                xb = torch.cat([xb, c], 1)
            outs.append(net(xb.to(dev)).cpu().numpy()[:, 0])
    return np.concatenate(outs, 0)


def run_dataset(name, paths, scale, net, params, dose_cond, sigma_cond,
                sigma, photons, dev, seed=2026):
    fovs = [load_image(p, scale) for p in paths]
    rng = np.random.default_rng(seed)
    records = []
    pair_list = []
    for fi, img in enumerate(fovs):
        for pi, (i, j, gt) in enumerate(patches(img, 128)):
            y = degrade(gt, sigma, photons, rng)
            pair_list.append(dict(fov=fi, patch=pi, i=i, j=j,
                                  gt=gt, y=y))
    ys = [r["y"] for r in pair_list]
    outs = apply_baselines_all(ys, params, sigma)
    outs["unet"] = unet_predict(net, ys, photons, sigma, dev,
                                dose_cond, sigma_cond)
    methods = list(outs)
    for k, r in enumerate(pair_list):
        row = dict(dataset=name, fov=r["fov"], patch=r["patch"],
                   i=r["i"], j=r["j"])
        for m in methods:
            p = outs[m][k]
            row[m] = dict(psnr=psnr(r["gt"], p), ssim=ssim(r["gt"], p),
                          fg=fg_psnr(r["gt"], p))
        records.append(row)
    return records, methods


def apply_baselines_all(ys, params, sigma):
    outs = {m: [] for m in ["tv", "wiener", "rl", "nlm"]}
    for y in ys:
        o = apply_baselines(y, params, sigma)
        for m in outs:
            outs[m].append(o[m])
    return outs


def fov_means(records, methods):
    keys = sorted(set((r["dataset"], r["fov"]) for r in records))
    out = {m: [] for m in methods}
    for d, f in keys:
        sel = [r for r in records if r["dataset"] == d and r["fov"] == f]
        for m in methods:
            out[m].append(float(np.mean([r[m]["psnr"] for r in sel])))
    return out


def boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = rng.choice(diff, len(diff), replace=True)
        vals.append(float(np.mean(s)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def compare(records, methods, ref="unet"):
    fm = fov_means(records, methods)
    net = np.asarray(fm[ref])
    stats = {}
    for m in methods:
        if m == ref:
            continue
        v = np.asarray(fm[m])
        d = net - v
        patch_d = np.asarray([r[ref]["psnr"] - r[m]["psnr"] for r in records])
        try:
            pw = float(wilcoxon(d).pvalue)
        except Exception:
            pw = float("nan")
        try:
            pp = float(wilcoxon(patch_d).pvalue)
        except Exception:
            pp = float("nan")
        stats[m] = dict(fov_mean_ref=float(net.mean()),
                        fov_mean_base=float(v.mean()),
                        diff=float(d.mean()), ci95=boot_ci(d),
                        p_fov=pw, p_patch=pp, n_fov=len(d),
                        n_patch=len(patch_d),
                        wins=int((d > 0).sum()))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unet_d300.pt")
    ap.add_argument("--dataset", default="bbbc014")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--photons", type=float, default=30.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--out", default="results/bench_clean.json")
    ap.add_argument("--external", default="",
                    help="comma-separated external dataset names")
    ap.add_argument("--external-limit", type=int, default=20,
                    help="max FOVs per external dataset")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = UNet(32, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    print("loaded", a.ckpt, "val", ck.get("val_psnr"), "in_ch", ck["in_ch"],
          "device", dev, flush=True)

    paths, scale = dataset_paths(a.dataset, a.channel)
    if a.dataset == "bbbc039":
        sp = bbbc039_official_split()
    else:
        sp = split_paths(paths, a.split_seed, a.n_val, a.n_test)
    val_fovs = [load_image(p, scale) for p in sp["val"]]
    params, val_scores = tune(val_fovs, a.sigma, a.photons)
    print("tuned", params, {k: round(v, 3) for k, v in val_scores.items()},
          flush=True)

    rec39, methods = run_dataset(a.dataset, sp["test"], scale, net,
                                 params, ck["dose_cond"], ck["sigma_cond"],
                                 a.sigma, a.photons, dev)
    stats39 = compare(rec39, methods)
    print(f"\n{a.dataset} held-out FOVs")
    for m, s in sorted(stats39.items(), key=lambda x: -x[1]["fov_mean_base"]):
        print(f"  {m:8s} {s['fov_mean_base']:.2f}  unet-base {s['diff']:+.2f} "
              f"CI {s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f} p_fov {s['p_fov']:.3g} "
              f"wins {s['wins']}/{s['n_fov']}", flush=True)
    for m in methods:
        print(f"  {m:8s}", round(float(np.mean(
            [r[m]["psnr"] for r in rec39])), 2),
            "ssim", round(float(np.mean([r[m]["ssim"] for r in rec39])), 4),
            "fg", round(float(np.mean([r[m]["fg"] for r in rec39])), 2), flush=True)

    out = dict(ckpt=a.ckpt, dataset=a.dataset, photons=a.photons,
               sigma=a.sigma, tuned=params, val_scores=val_scores,
               main=dict(records=rec39, stats=stats39), methods=methods)
    for ext in [x for x in a.external.split(",") if x]:
        epaths, escale = dataset_paths(ext)
        if a.external_limit and len(epaths) > a.external_limit:
            step = max(1, len(epaths) // a.external_limit)
            epaths = epaths[::step][:a.external_limit]
        rec, _ = run_dataset(ext, epaths, escale, net, params,
                             ck["dose_cond"], ck["sigma_cond"],
                             a.sigma, a.photons, dev, seed=2027)
        st = compare(rec, methods)
        print(f"\n{ext} external FOVs (no training on this dataset)")
        for m, s in sorted(st.items(), key=lambda x: -x[1]["fov_mean_base"]):
            print(f"  {m:8s} {s['fov_mean_base']:.2f}  unet-base {s['diff']:+.2f} "
                  f"CI {s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f} p_fov {s['p_fov']:.3g}", flush=True)
        out[ext] = dict(records=rec, stats=st)
    save_json(a.out, out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
