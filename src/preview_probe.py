"""Probe preview doses and target definitions for adaptive allocation."""
from __future__ import annotations

import json
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
from adaptive_eval import (fov_features, load_nets, required_doses)
from common import dataset_paths, load_image, split_paths
from analyze_adaptive import (evaluate, knn_fit, knn_loo, knn_pred, loo_ridge,
                              ridge_fit, ridge_pred)


def fixed_dose_idx(P, doses, target, success=0.9):
    for j in range(len(doses)):
        if evaluate(P, [j] * len(P), doses, target)[1] >= success:
            return j
    return len(doses) - 1


def policy_sweep(Pv, Pt, Fv, Ft, doses, target, success=0.9):
    yv = np.log10([doses[i] for i in required_doses(Pv, target)])
    out = {}
    for name, cols in [("unc", list(range(6, 11))),
                       ("stats", list(range(6))),
                       ("all", list(range(11)))]:
        loo = loo_ridge(Fv[:, cols], yv)
        resid = yv - loo
        model = ridge_fit(Fv[:, cols], yv)
        pred = ridge_pred(model, Ft[:, cols])
        best = None
        for q in np.linspace(0.5, 0.999, 60):
            margin = float(np.quantile(resid, q))
            idx = [int(np.argmin(np.abs(np.log10(doses) - (pred[i] + margin))))
                   for i in range(len(pred))]
            _, sv = evaluate(Pv, idx, doses, target)
            if sv >= success:
                best = (margin, idx, sv)
                break
        if best is None:
            out[f"ridge_{name}"] = dict(dose=float(doses[-1]), success=0.0)
            continue
        margin, _, _ = best
        idx = [int(np.argmin(np.abs(np.log10(doses) - (pred[i] + margin))))
               for i in range(len(pred))]
        c, s = evaluate(Pt, idx, doses, target)
        out[f"ridge_{name}"] = dict(dose=c, success=s, margin=margin)
        # kNN with max-of-neighbours as conservative estimate
        bk = None
        for k in [3, 5, 8]:
            pv = knn_loo(Fv[:, cols], yv, k=k, q=1.0)
            idxv = [int(np.argmin(np.abs(np.log10(doses) - pv[i])))
                    for i in range(len(pv))]
            _, sv = evaluate(Pv, idxv, doses, target)
            if sv >= success:
                bk = k
                break
        if bk:
            m = knn_fit(Fv[:, cols], yv)
            pv = knn_pred(m, Ft[:, cols], k=bk, q=1.0)
            idx = [int(np.argmin(np.abs(np.log10(doses) - pv[i])))
                   for i in range(len(pv))]
            c, s = evaluate(Pt, idx, doses, target)
            out[f"knn_{name}"] = dict(dose=c, success=s, k=bk)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/adaptive_bbbc014.json"
    d = json.load(open(path))
    doses = d["doses"]
    Pv = np.asarray(d["val_curves"])
    Pt = np.asarray(d["test_curves"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = load_nets(["checkpoints/bbbc014_dcond.pt",
                      "checkpoints/bbbc014_dcond_s1.pt",
                      "checkpoints/bbbc014_dcond_s2.pt"], dev)
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    val = [load_image(p, scale) for p in sp["val"]]
    test = [load_image(p, scale) for p in sp["test"]]

    for preview in [3.0, 10.0, 30.0]:
        Fv = fov_features(nets, val, preview, dev)
        Ft = fov_features(nets, test, preview, dev)
        print(f"\n=== preview {preview:g} photons")
        for T in [34.0, 35.0, 35.5, 36.0, 36.5]:
            fixed = fixed_dose_idx(Pv, doses, T)
            oc = float(np.mean([doses[j] for j in required_doses(Pt, T)]))
            res = policy_sweep(Pv, Pt, Fv, Ft, doses, T)
            line = f"T {T:.1f} fixed {doses[fixed]:g} oracle {oc:.0f} " \
                   f"save {100*(1-oc/doses[fixed]):.0f}% |"
            for k, v in res.items():
                line += f" {k} {v['dose']:.0f}({v['success']:.2f})"
            print(line, flush=True)


if __name__ == "__main__":
    main()
