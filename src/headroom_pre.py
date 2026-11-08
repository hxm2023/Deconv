"""Headroom for per-FOV adaptive photon allocation (classical TV probe).

We compute per-FOV PSNR-vs-dose curves on validation and test FOVs, then ask:
how much dose would an oracle per-FOV allocator save over the best fixed dose
that reaches a target success rate? A preview-feature rule is also fitted to
see whether a simple observable rule already captures the oracle gain.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv
from common import (bbbc039_official_split, dataset_paths, degrade, load_image,
                    patches, psnr, save_json, split_paths)

WEIGHTS = [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4, 0.6]


def sample_patches(img, n=5):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def curve_for(img, dose, weight, seeds):
    vals = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for gt in sample_patches(img):
            y = degrade(gt, 2.0, dose, rng)
            vals.append(psnr(gt, tv(y, weight)))
    return float(np.mean(vals))


def tune_weights(val_fovs, doses, seeds):
    best = {}
    for d in doses:
        scores = []
        for w in WEIGHTS:
            v = [curve_for(img, d, w, seeds) for img in val_fovs]
            scores.append(float(np.mean(v)))
        best[d] = WEIGHTS[int(np.argmax(scores))]
        print(f"  dose {d}: TV weight {best[d]} (val {max(scores):.2f})", flush=True)
    return best


def curves(fovs, doses, weights, seeds):
    out = []
    for img in fovs:
        row = {}
        for d in doses:
            row[d] = curve_for(img, d, weights[d], seeds)
        out.append(row)
    return out


def preview_features(img, dose, seed=101):
    """Observable features from a low-dose preview (no GT)."""
    rng = np.random.default_rng(seed)
    feats = []
    for gt in sample_patches(img):
        y = degrade(gt, 2.0, dose, rng)
        gy, gx = np.gradient(y)
        g = np.hypot(gx, gy)
        lap = np.abs(4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1]
                     - y[1:-1, :-2] - y[1:-1, 2:])
        feats.append([y.mean(), y.std(), np.quantile(y, 0.99),
                      g.mean(), np.median(lap), np.quantile(y, 0.5)])
    return np.mean(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bbbc014")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--doses", default="3,10,30,100,300")
    ap.add_argument("--seeds", default="11,12,13")
    ap.add_argument("--out", default="results/headroom_tv.json")
    a = ap.parse_args()
    doses = [float(x) for x in a.doses.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    paths, scale = dataset_paths(a.dataset, a.channel)
    if a.dataset == "bbbc039":
        sp = bbbc039_official_split()
    else:
        sp = split_paths(paths, a.split_seed, a.n_val, a.n_test)
    val = [load_image(p, scale) for p in sp["val"]]
    test = [load_image(p, scale) for p in sp["test"]]
    print(f"{a.dataset}: {len(sp['train'])} train / {len(val)} val / "
          f"{len(test)} test FOVs", flush=True)

    print("tuning TV weight per dose on validation FOVs", flush=True)
    weights = tune_weights(val, doses, seeds)
    cv = curves(val, doses, weights, seeds)
    ct = curves(test, doses, weights, seeds)
    save_json(a.out, dict(dataset=a.dataset, doses=doses, seeds=seeds,
                          weights={str(k): v for k, v in weights.items()},
                          val=cv, test=ct))

    print("\nper-dose test PSNR: min / p10 / median / max")
    for d in doses:
        v = np.array([r[d] for r in ct])
        print(f"  {d:5.0f}: {v.min():.2f} / {np.percentile(v,10):.2f} / "
              f"{np.median(v):.2f} / {v.max():.2f}", flush=True)

    # Target: 10th percentile at the maximum dose, so 90% succeed at max dose.
    dmax = doses[-1]
    target = float(np.percentile([r[dmax] for r in cv], 10))
    fixed = None
    for d in doses:
        if np.mean([r[d] >= target for r in cv]) >= 0.9:
            fixed = d
            break
    print(f"\ntarget {target:.2f} (val p10 at {dmax:g}); "
          f"fixed dose {fixed}", flush=True)
    if fixed is None:
        return
    costs = []
    for r in ct:
        c = next((d for d in doses if r[d] >= target), dmax)
        costs.append(c)
    oracle = float(np.mean(costs))
    succ_fixed = float(np.mean([r[fixed] >= target for r in ct]))
    print(f"test fixed={fixed:g} success {succ_fixed:.2f}; "
          f"oracle cost {oracle:.1f} -> save {100*(1-oracle/fixed):.1f}%", flush=True)

    # Observable preview rule fitted on validation FOVs.
    Xv = np.array([preview_features(img, doses[0]) for img in val])
    Xt = np.array([preview_features(img, doses[0]) for img in test])
    yv = np.array([np.log10(next((d for d in doses if r[d] >= target), dmax))
                   for r in cv])
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
    Xv = (Xv - mu) / sd
    Xt = (Xt - mu) / sd
    Xv1 = np.c_[np.ones(len(Xv)), Xv]
    Xt1 = np.c_[np.ones(len(Xt)), Xt]
    coef = np.linalg.solve(Xv1.T @ Xv1 + 1e-2 * np.eye(Xv1.shape[1]),
                           Xv1.T @ yv)
    logpred = np.clip(Xt1 @ coef, np.log10(doses[0]), np.log10(doses[-1]))
    pred = 10 ** logpred
    costs_p, succ_p = [], []
    for r, d in zip(ct, pred):
        d = min(doses, key=lambda z: abs(z - d))
        costs_p.append(d)
        succ_p.append(r[d] >= target)
    print(f"observable preview rule: dose {np.mean(costs_p):.1f} "
          f"(fixed {fixed:g}), success {np.mean(succ_p):.2f}, "
          f"save {100*(1-np.mean(costs_p)/fixed):.1f}%", flush=True)
    save_json(a.out, dict(dataset=a.dataset, doses=doses, seeds=seeds,
                          weights={str(k): v for k, v in weights.items()},
                          val=cv, test=ct, target=target, fixed_dose=fixed,
                          oracle_dose=oracle, oracle_success=succ_fixed,
                          oracle_save=100 * (1 - oracle / fixed),
                          preview_dose=float(np.mean(costs_p)),
                          preview_success=float(np.mean(succ_p)),
                          preview_save=100 * (1 - np.mean(costs_p) / fixed),
                          preview_coef=coef.tolist()))


if __name__ == "__main__":
    main()
