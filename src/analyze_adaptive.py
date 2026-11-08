"""Offline analysis of adaptive allocation policies across targets.

Loads the curves/features saved by adaptive_eval.py and compares:
- fixed dose (selected on validation)
- oracle per-FOV dose
- ridge regression on preview features
- k-nearest-neighbour calibration
- binned calibration
All policies are fitted on validation FOVs only; test FOVs are untouched.
"""
from __future__ import annotations

import json
import sys

import numpy as np


NAMES = ["mean", "std", "med", "q99", "grad", "lap",
         "u_mean", "u_p90", "u_max", "u_std", "u_p99"]


def required(P, target):
    mean = P.mean(2)
    idx = np.full(len(P), P.shape[1] - 1)
    for i in range(len(P)):
        for j in range(P.shape[1]):
            if mean[i, j] >= target:
                idx[i] = j
                break
    return idx


def ridge_fit(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.c_[np.ones(len(Z)), Z]
    coef = np.linalg.solve(Z1.T @ Z1 + alpha * np.eye(Z1.shape[1]), Z1.T @ y)
    return dict(mu=mu, sd=sd, coef=coef)


def ridge_pred(m, X):
    Z = (X - m["mu"]) / m["sd"]
    return np.c_[np.ones(len(Z)), Z] @ m["coef"]


def loo_ridge(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        p[i] = ridge_pred(ridge_fit(X[m], y[m]), X[i:i + 1])[0]
    return p


def knn_fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return dict(mu=mu, sd=sd, X=(X - mu) / sd, y=y)


def knn_pred(m, X, k=5, q=0.9):
    Z = (X - m["mu"]) / m["sd"]
    out = []
    for z in Z:
        d = np.linalg.norm(m["X"] - z, axis=1)
        nn = np.argsort(d)[:k]
        out.append(float(np.quantile(m["y"][nn], q)))
    return np.asarray(out)


def knn_loo(X, y, k=5, q=0.9):
    # light leave-one-out: exclude self from neighbours
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    out = np.zeros(len(y))
    for i in range(len(y)):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = np.inf
        nn = np.argsort(d)[:k]
        out[i] = float(np.quantile(y[nn], q))
    return out


def binned_fit(X, y, col, nbin=4):
    x = X[:, col]
    edges = np.quantile(x, np.linspace(0, 1, nbin + 1))
    return dict(col=col, edges=edges, y=y)


def binned_pred(m, X, q=0.9):
    out = []
    for x in X[:, m["col"]]:
        b = int(np.clip(np.searchsorted(m["edges"], x, side="right") - 1,
                        0, len(m["edges"]) - 2))
        lo, hi = m["edges"][b], m["edges"][b + 1]
        mask = (m["y"]["x"] >= lo) & (m["y"]["x"] <= hi)
        vals = m["y"]["y"][mask]
        out.append(float(np.quantile(vals, q)) if len(vals) else float(np.median(m["y"]["y"])))
    return np.asarray(out)


def evaluate(P, dose_idx, doses, target, seeds=None):
    n_fov, n_dose, n_seed = P.shape
    succ = float(np.mean([P[i, dose_idx[i], s] >= target
                          for i in range(n_fov) for s in range(n_seed)]))
    cost = float(np.mean([doses[j] for j in dose_idx]))
    return cost, succ


def choose_margin(pred_log, resid, P, doses, target, success):
    """Smallest margin (from val LOO residuals) with val success >= target."""
    best = None
    for q in np.linspace(0.5, 0.99, 25):
        margin = float(np.quantile(resid, q))
        idx = [int(np.argmin(np.abs(np.log10(doses) - (pred_log[i] + margin))))
               for i in range(len(pred_log))]
        _, s = evaluate(P, idx, doses, target)
        if s >= success:
            best = (margin, idx, s)
            break
    if best is None:
        idx = [len(doses) - 1] * len(pred_log)
        best = (float(np.max(resid)), idx, 1.0)
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/adaptive_bbbc014.json"
    d = json.load(open(path))
    doses = d["doses"]
    Pv = np.asarray(d["val_curves"])
    Pt = np.asarray(d["test_curves"])
    Fv = np.asarray(d["features_val"])
    Ft = np.asarray(d["features_test"])
    cols_unc = list(range(6, 11))
    cols_stats = list(range(6))
    targets = [33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5]

    print("target fixed oracle | policy: dose succ save")
    out = dict(targets=targets, fixed=[], oracle=[], policies={})
    for T in targets:
        rv = required(Pv, T)
        rt = required(Pt, T)
        fixed = None
        for j in range(len(doses)):
            _, s = evaluate(Pv, [j] * len(Pv), doses, T)
            if s >= 0.9:
                fixed = j
                break
        if fixed is None:
            print(f"{T:.2f}: no fixed dose")
            continue
        fc, fs = evaluate(Pt, [fixed] * len(Pt), doses, T)
        oc, os_ = evaluate(Pt, rt, doses, T)
        out["fixed"].append(dict(target=T, dose=fc, success=fs))
        out["oracle"].append(dict(target=T, dose=oc, success=os_))
        line = f"{T:.2f} fixed {doses[fixed]:.0f}({fs:.2f}) oracle {oc:.0f}({os_:.2f}) save {100*(1-oc/doses[fixed]):.0f}% |"
        yv = np.log10([doses[i] for i in rv])
        for name, cols in [("unc", cols_unc), ("stats", cols_stats),
                           ("all", list(range(11)))]:
            # ridge
            loo = loo_ridge(Fv[:, cols], yv)
            resid = yv - loo
            model = ridge_fit(Fv[:, cols], yv)
            pred = ridge_pred(model, Ft[:, cols])
            margin, idx, vs = choose_margin(pred, resid, Pv, doses, T, 0.9)
            idx_t = [int(np.argmin(np.abs(np.log10(doses) - (pred[i] + margin))))
                     for i in range(len(pred))]
            tc, ts = evaluate(Pt, idx_t, doses, T)
            out["policies"].setdefault(f"ridge_{name}", []).append(
                dict(target=T, dose=tc, success=ts, margin=float(margin)))
            line += f" ridge-{name} {tc:.0f}({ts:.2f})"
            # knn on the same features, q from val LOO
            bestk = None
            for k in [3, 5, 8]:
                for q in [0.75, 0.9, 0.95]:
                    pv = knn_loo(Fv[:, cols], yv, k=k, q=q)
                    _, s = evaluate(Pv, [int(np.argmin(np.abs(np.log10(doses) - pv[i])))
                                         for i in range(len(pv))], doses, T)
                    if s >= 0.9 and bestk is None:
                        bestk = (k, q)
            if bestk:
                m = knn_fit(Fv[:, cols], yv)
                pv = knn_pred(m, Ft[:, cols], k=bestk[0], q=bestk[1])
                idx_t = [int(np.argmin(np.abs(np.log10(doses) - pv[i])))
                         for i in range(len(pv))]
                tc, ts = evaluate(Pt, idx_t, doses, T)
                out["policies"].setdefault(f"knn_{name}", []).append(
                    dict(target=T, dose=tc, success=ts, k=bestk[0], q=bestk[1]))
                line += f" knn-{name} {tc:.0f}({ts:.2f})"
        print(line)
    json.dump(out, open("results/adaptive_policy_analysis.json", "w"),
              indent=2)

    def interp(points, level):
        pts = sorted((p["success"], p["dose"]) for p in points)
        if level <= pts[0][0]:
            return pts[0][1]
        if level >= pts[-1][0]:
            return pts[-1][1]
        for k in range(1, len(pts)):
            if pts[k - 1][0] <= level <= pts[k][0]:
                x0, y0 = pts[k - 1]
                x1, y1 = pts[k]
                if x1 == x0:
                    return min(y0, y1)
                return y0 + (level - x0) / (x1 - x0) * (y1 - y0)
        return None

    print("\nmatched-success photon cost")
    for level in [0.90, 0.95]:
        f = interp(out["fixed"], level)
        o = interp(out["oracle"], level)
        line = f"  success {level:.2f}: fixed {f:.0f} oracle {o:.0f}"
        for name, pts in out["policies"].items():
            v = interp(pts, level)
            line += f" | {name} {v:.0f} ({100*(1-v/f):.0f}%)"
        print(line)


if __name__ == "__main__":
    main()
