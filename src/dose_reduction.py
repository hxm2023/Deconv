"""Photon dose needed to reach a target PSNR: U-Net vs tuned TV."""
from __future__ import annotations

import json
import sys

import numpy as np


def load_curves(path, key, valkey):
    d = json.load(open(path))
    doses = [float(x) for x in d["doses"]]
    rows = d[key]
    out = []
    for r in rows:
        vals = []
        for dose in doses:
            k = str(float(dose))
            v = r[k]
            vals.append(v[valkey] if isinstance(v, dict) else v)
        out.append(vals)
    return np.asarray(doses), np.asarray(out)


def dose_for(doses, psnrs, T):
    if psnrs[0] >= T:
        return float(doses[0])
    for k in range(1, len(doses)):
        if psnrs[k] >= T:
            x0, x1 = np.log10(doses[k - 1]), np.log10(doses[k])
            y0, y1 = psnrs[k - 1], psnrs[k]
            if y1 <= y0:
                return float(doses[k])
            return float(10 ** (x0 + (T - y0) / (y1 - y0) * (x1 - x0)))
    return None


def main():
    u = json.load(open("results/unet_dose_response.json"))
    du = np.array([float(x) for x in u["doses"]])
    nf = len(u["test"][str(float(du[0]))])
    Pu = np.array([[u["test"][str(float(d))][i]["fov_mean_psnr"]
                    for d in du] for i in range(nf)])
    h = json.load(open("results/headroom_bbbc014_2020.json"))
    dt = np.array([float(x) for x in h["doses"]])
    Pt = np.array([[h["test"][i][str(float(d))] for d in dt]
                   for i in range(len(h["test"]))])
    print("U-Net doses", du.tolist())
    print("TV doses", dt.tolist())
    targets = [28, 30, 32, 34, 35, 36, 37, 38]
    out = dict(targets=targets, rows=[])
    print("target  U-Net dose  TV dose  ratio  n_both  TV reach  U-Net reach")
    for T in targets:
        u = np.array([dose_for(du, p, T) if p is not None else np.nan
                      for p in Pu], dtype=float)
        t = np.array([dose_for(dt, p, T) if p is not None else np.nan
                      for p in Pt], dtype=float)
        # dose_for returns None if unreachable
        u = np.array([np.nan if v is None else v for v in u], dtype=float)
        t = np.array([np.nan if v is None else v for v in t], dtype=float)
        m = ~np.isnan(u) & ~np.isnan(t)
        ratio = float(np.median(t[m] / u[m])) if m.any() else float("nan")
        row = dict(target=T, unet_dose=float(np.nanmedian(u)),
                   tv_dose=float(np.nanmedian(t)), ratio=ratio,
                   n_both=int(m.sum()), tv_reach=float(np.mean(~np.isnan(t))),
                   unet_reach=float(np.mean(~np.isnan(u))))
        out["rows"].append(row)
        print(f"{T:6.1f} {row['unet_dose']:10.2f} {row['tv_dose']:9.2f} "
              f"{ratio:7.2f} {m.sum():7d} {row['tv_reach']:8.2f} "
              f"{row['unet_reach']:10.2f}")
    json.dump(out, open("results/dose_reduction.json", "w"), indent=2)


if __name__ == "__main__":
    main()
