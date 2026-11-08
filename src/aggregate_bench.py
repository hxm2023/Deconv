"""Aggregate per-dose benchmarks into dose-response tables and dose savings."""
from __future__ import annotations

import json

import numpy as np

DOSES = [3, 10, 30, 100, 300]
METHODS = ["tv", "nlm", "wiener", "rl", "unet"]
METRICS = ["psnr", "ssim", "fg"]


def load_dose(d):
    j = json.load(open(f"results/bench_bbbc014_d{d}.json"))
    recs = j["main"]["records"]
    fovs = sorted(set(r["fov"] for r in recs))
    out = {m: [] for m in METHODS}
    for f in fovs:
        sel = [r for r in recs if r["fov"] == f]
        for m in METHODS:
            out[m].append([float(np.mean([r[m][k] for r in sel])) for k in METRICS])
    return j, {m: np.asarray(v) for m, v in out.items()}


def dose_for(doses, vals, target):
    if vals[0] >= target:
        return float(doses[0])
    for k in range(1, len(doses)):
        if vals[k] >= target:
            x0, x1 = np.log10(doses[k - 1]), np.log10(doses[k])
            y0, y1 = vals[k - 1], vals[k]
            if y1 <= y0:
                return float(doses[k])
            return float(10 ** (x0 + (target - y0) / (y1 - y0) * (x1 - x0)))
    return None


def main():
    data = {d: load_dose(d) for d in DOSES}
    n_fov = len(next(iter(data[3][1]["tv"])))
    print("n FOV", n_fov)
    summary = dict(doses=DOSES, methods=METHODS, metrics=METRICS, fov={},
                   dose_savings={})
    print("dose method PSNR SSIM FG")
    for d in DOSES:
        for m in METHODS:
            v = data[d][1][m]
            print(f"{d:4d} {m:7s} {v[:,0].mean():6.2f} {v[:,1].mean():6.4f} "
                  f"{v[:,2].mean():6.2f}")
            summary["fov"].setdefault(m, {})[str(d)] = v.tolist()

    print("\nmatched-quality dose (median over FOVs)")
    ssim_targets = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]
    fg_targets = [22, 24, 26, 28, 30, 32]
    psnr_targets = [32, 33, 34, 35, 36, 37, 38]
    for metric, targets in [("ssim", ssim_targets), ("fg", fg_targets),
                            ("psnr", psnr_targets)]:
        mi = METRICS.index(metric)
        print(f"\n{metric} target  tv  nlm  wiener  rl  unet  unet/tv")
        for T in targets:
            row = {}
            for m in METHODS:
                vals = [data[d][1][m][:, mi] for d in DOSES]
                out = []
                for i in range(n_fov):
                    out.append(dose_for(DOSES, [v[i] for v in vals], T))
                arr = np.array([np.nan if x is None else x for x in out])
                row[m] = float(np.nanmedian(arr))
            ratio = row["tv"] / row["unet"] if row["unet"] else np.nan
            print(f"{T:7.2f} " + " ".join(f"{row[m]:6.2f}" for m in METHODS) +
                  f" {ratio:7.2f}")
            summary["dose_savings"].setdefault(metric, []).append(
                dict(target=T, **{m: row[m] for m in METHODS},
                     unet_vs_tv=ratio))
    json.dump(summary, open("results/bench_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
