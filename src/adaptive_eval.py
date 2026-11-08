"""Uncertainty-guided per-FOV photon allocation for low-dose deconvolution.

Protocol
--------
1. A dose-conditioned U-Net ensemble reconstructs a FOV from any photon dose.
2. At a cheap preview dose, ensemble disagreement (predictive uncertainty) and
   GT-free image statistics are computed.
3. A ridge model predicts the dose needed to reach a target reconstruction
   quality; a conformal margin calibrated on validation FOVs enforces the
   target success rate.
4. Test FOVs are never used for training, target selection, or calibration.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (bbbc039_official_split, dataset_paths, degrade, psnr,
                    load_image, patches, save_json, split_paths)
from model import UNet

DOSE_MIN, DOSE_MAX = 3.0, 300.0


def cond(photons):
    lo, hi = math.log10(DOSE_MIN), math.log10(DOSE_MAX)
    return (math.log10(photons) - lo) / (hi - lo)


def load_nets(ckpts, dev):
    nets = []
    for p in ckpts:
        ck = torch.load(p, map_location=dev, weights_only=False)
        net = UNet(32, ck["in_ch"]).to(dev)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        nets.append(net)
    return nets


def ensemble_predict(nets, ys, photons, dev, dose_cond=True, batch=128):
    """Return ensemble mean and variance, each [N, H, W]."""
    means, vars_ = [], []
    with torch.no_grad():
        for q in range(0, len(ys), batch):
            xb0 = torch.as_tensor(np.asarray(ys[q:q + batch]),
                                  dtype=torch.float32)[:, None]
            if dose_cond:
                h, w = xb0.shape[-2:]
                c = torch.full((len(xb0), 1, h, w), cond(photons),
                               dtype=torch.float32)
                xb = torch.cat([xb0, c], 1).to(dev)
            else:
                xb = xb0.to(dev)
            preds = torch.stack([n(xb)[:, 0] for n in nets], 0)
            means.append(preds.mean(0).cpu().numpy())
            vars_.append(preds.var(0, unbiased=False).cpu().numpy())
    return np.concatenate(means, 0), np.concatenate(vars_, 0)


def image_stats(ys):
    """GT-free statistics of a noisy FOV (averaged over patches)."""
    rows = []
    for y in ys:
        gy, gx = np.gradient(y)
        g = np.hypot(gx, gy)
        lap = np.abs(4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1]
                     - y[1:-1, :-2] - y[1:-1, 2:])
        rows.append([y.mean(), y.std(), np.quantile(y, 0.5),
                     np.quantile(y, 0.99), g.mean(), np.median(lap)])
    return np.mean(rows, 0)


def sample_patches(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return ps[::step][:n]


def fov_curves(nets, fovs, doses, seeds, dev):
    """Per-FOV mean PSNR for each dose and noise seed."""
    P = np.zeros((len(fovs), len(doses), len(seeds)))
    for fi, img in enumerate(fovs):
        ps = sample_patches(img)
        gts = [gt for _, _, gt in ps]
        for di, d in enumerate(doses):
            for si, sd in enumerate(seeds):
                rng = np.random.default_rng(sd)
                ys = [degrade(gt, 2.0, d, rng) for gt in gts]
                mean, _ = ensemble_predict(nets, ys, d, dev)
                P[fi, di, si] = np.mean(
                    [psnr(gt, p) for gt, p in zip(gts, mean)])
        print(f"  FOV {fi+1}/{len(fovs)}", flush=True)
    return P


def fov_features(nets, fovs, dose, dev, seed=777):
    feats = []
    for img in fovs:
        ps = sample_patches(img)
        gts = [gt for _, _, gt in ps]
        rng = np.random.default_rng(seed)
        ys = [degrade(gt, 2.0, dose, rng) for gt in gts]
        mean, var = ensemble_predict(nets, ys, dose, dev)
        u = np.sqrt(np.clip(var, 0, None))
        stat = image_stats(ys)
        feats.append(np.r_[stat,
                           u.mean(), np.quantile(u, 0.9), u.max(),
                           u.std(), np.quantile(u, 0.99)])
    return np.asarray(feats)


def ridge_fit(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.c_[np.ones(len(Z)), Z]
    coef = np.linalg.solve(Z1.T @ Z1 + alpha * np.eye(Z1.shape[1]),
                           Z1.T @ y)
    return dict(mu=mu, sd=sd, coef=coef)


def ridge_predict(model, X):
    Z = (X - model["mu"]) / model["sd"]
    return np.c_[np.ones(len(Z)), Z] @ model["coef"]


def required_doses(P, target):
    """First dose whose mean-over-seeds PSNR reaches target."""
    n_fov, n_dose, n_seed = P.shape
    mean = P.mean(2)
    out = np.full(n_fov, n_dose - 1)
    for i in range(n_fov):
        for j in range(n_dose):
            if mean[i, j] >= target:
                out[i] = j
                break
    return out


def loo_required(model_fn, X, y):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        model = model_fn(X[m], y[m])
        pred[i] = ridge_predict(model, X[i:i + 1])[0]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True,
                    help="comma-separated dose-conditioned checkpoints")
    ap.add_argument("--dataset", default="bbbc014")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--doses", default="3,5,10,20,30,50,100,200,300")
    ap.add_argument("--curve-seeds", default="31,32,33")
    ap.add_argument("--episode-seeds", default="41,42,43,44,45")
    ap.add_argument("--preview", type=float, default=3.0)
    ap.add_argument("--target-q", type=float, default=0.10)
    ap.add_argument("--success", type=float, default=0.9)
    ap.add_argument("--out", default="results/adaptive_bbbc014.json")
    a = ap.parse_args()
    doses = [float(x) for x in a.doses.split(",")]
    curve_seeds = [int(x) for x in a.curve_seeds.split(",")]
    ep_seeds = [int(x) for x in a.episode_seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = load_nets(a.ckpts.split(","), dev)
    print(f"ensemble of {len(nets)} models on {dev}", flush=True)

    paths, scale = dataset_paths(a.dataset, a.channel)
    if a.dataset == "bbbc039":
        sp = bbbc039_official_split()
    else:
        sp = split_paths(paths, a.split_seed, a.n_val, a.n_test)
    val = [load_image(p, scale) for p in sp["val"]]
    test = [load_image(p, scale) for p in sp["test"]]
    print(f"val {len(val)} test {len(test)} FOVs", flush=True)

    print("validation dose curves", flush=True)
    Pv = fov_curves(nets, val, doses, curve_seeds, dev)
    print("test dose curves", flush=True)
    Pt = fov_curves(nets, test, doses, curve_seeds, dev)

    target = float(np.quantile(Pv[:, -1, :].mean(1), a.target_q,
                               method="lower"))
    fixed_idx = None
    for j, d in enumerate(doses):
        succ = float((Pv[:, j, :] >= target).mean())
        if succ >= a.success:
            fixed_idx = j
            break
    if fixed_idx is None:
        fixed_idx = len(doses) - 1
    fixed_dose = doses[fixed_idx]
    fixed_succ = float((Pt[:, fixed_idx, :] >= target).mean())
    oracle_idx = required_doses(Pt, target)
    oracle_cost = float(np.mean([doses[j] for j in oracle_idx]))
    oracle_succ = float(np.mean([Pt[i, oracle_idx[i], s] >= target
                                 for i in range(len(test))
                                 for s in range(len(curve_seeds))]))
    req_v = required_doses(Pv, target)
    print(f"target {target:.3f}; fixed {fixed_dose:g} (test succ {fixed_succ:.3f}); "
          f"oracle cost {oracle_cost:.1f} succ {oracle_succ:.3f}", flush=True)

    Fv = fov_features(nets, val, a.preview, dev)
    Ft = fov_features(nets, test, a.preview, dev)
    yv = np.log10([doses[j] for j in req_v])
    # features: first 6 are image statistics; last 5 are uncertainty
    policy_results = {}
    for name, cols in [("brightness", [0]), ("simple+unc", [0, 1, 6]),
                       ("stats", list(range(6))), ("stats+unc", list(range(Fv.shape[1])))]:
        loo = loo_required(lambda X, y: ridge_fit(X, y), Fv[:, cols], yv)
        resid = yv - loo
        model = ridge_fit(Fv[:, cols], yv)
        pred_t = ridge_predict(model, Ft[:, cols])
        print(f"\npolicy {name}: LOO residual p50 {np.median(resid):.3f} "
              f"p90 {np.quantile(resid, 0.9):.3f}", flush=True)
        policy_results[name] = dict(loo_residual_p50=float(np.median(resid)),
                                    loo_residual_p90=float(np.quantile(resid, 0.9)),
                                    margins={})
        for q in [0.5, 0.75, 0.9, 0.95]:
            margin = float(np.quantile(resid, q))
            idx = [int(np.argmin(np.abs(np.log10(doses) -
                                       (pred_t[i] + margin))))
                   for i in range(len(test))]
            costs = [doses[j] for j in idx]
            succ = float(np.mean([Pt[i, idx[i], s] >= target
                                  for i in range(len(test))
                                  for s in range(len(curve_seeds))]))
            print(f"  margin q={q:.2f} ({10**margin:.2f}x): dose "
                  f"{np.mean(costs):.1f} succ {succ:.3f} "
                  f"save {100*(1-np.mean(costs)/fixed_dose):.1f}%", flush=True)
            policy_results[name]["margins"][str(q)] = dict(
                margin=margin, dose=float(np.mean(costs)),
                success=succ,
                save=100 * (1 - float(np.mean(costs)) / fixed_dose),
                chosen=[float(c) for c in costs])

    save_json(a.out, dict(
        ckpts=a.ckpts, dataset=a.dataset, doses=doses,
        curve_seeds=curve_seeds, target=target, fixed_dose=fixed_dose,
        fixed_success=fixed_succ, oracle_dose=oracle_cost,
        oracle_success=oracle_succ,
        val_curves=Pv.tolist(), test_curves=Pt.tolist(),
        required_val=req_v.tolist(), required_test=oracle_idx.tolist(),
        features_val=Fv.tolist(), features_test=Ft.tolist(),
        policies=policy_results,
        note="features last 5 columns are ensemble uncertainty"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
