"""Two-stage photon allocation: cheap preview + re-acquire uncertain FOVs.

All FOVs get a preview dose p. FOVs ranked as risky by ensemble uncertainty
(or a simple image statistic) are re-acquired at dose D. The comparison is a
fixed dose that reaches the same validation success rate.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_eval import fov_features, load_nets
from common import dataset_paths, load_image, save_json, split_paths


def fixed_dose(P, doses, T, level=0.9):
    for j in range(len(doses)):
        if float((P[:, j, :] >= T).mean()) >= level:
            return j
    return len(doses) - 1


def stage_success(P, p_idx, D_idx, flag, T):
    n, _, ns = P.shape
    return float(np.mean([(P[i, D_idx, s] if flag[i] else P[i, p_idx, s]) >= T
                          for i in range(n) for s in range(ns)]))


def choose_k(Pv, p_idx, D_idx, score, T, level):
    order = np.argsort(-score)
    for k in range(len(score) + 1):
        flag = np.zeros(len(score), bool)
        flag[order[:k]] = True
        if stage_success(Pv, p_idx, D_idx, flag, T) >= level:
            return k, flag
    return len(score), np.ones(len(score), bool)


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

    out = []
    for preview in [3.0, 10.0, 30.0]:
        Fv = fov_features(nets, val, preview, dev)
        Ft = fov_features(nets, test, preview, dev)
        p_idx = doses.index(preview)
        print(f"\n=== preview {preview:g}")
        for T in [34.0, 35.0, 36.0, 37.0]:
            fv = fixed_dose(Pv, doses, T, 0.9)
            fcost = doses[fv]
            fsucc = float((Pt[:, fv, :] >= T).mean())
            line = f"T {T:.1f} fixed {fcost:g} test_succ {fsucc:.2f} |"
            for D in [30.0, 50.0, 100.0, 200.0]:
                if D <= preview:
                    continue
                D_idx = doses.index(D)
                for name, idx in [("unc", 9), ("std", 1), ("mean", 0)]:
                    k, _ = choose_k(Pv, p_idx, D_idx, Fv[:, idx], T, 0.9)
                    order = np.argsort(-Ft[:, idx])
                    flag = np.zeros(len(Ft), bool)
                    flag[order[:k]] = True
                    succ = stage_success(Pt, p_idx, D_idx, flag, T)
                    cost = preview + (k / len(Ft)) * (D - preview)
                    line += f" D{D:g}{name} k{k} {cost:.1f}({succ:.2f})"
                    out.append(dict(preview=preview, target=T, D=D,
                                    score=name, k=k, cost=cost, success=succ,
                                    fixed_dose=fcost, fixed_success=fsucc))
            print(line, flush=True)
    save_json("results/two_stage.json", dict(rows=out))


if __name__ == "__main__":
    main()
