"""Fixed-dose protocol: smallest dose with >=90% of FOVs above a target."""
from __future__ import annotations

import json

import numpy as np

DOSES = [3, 10, 30, 100, 300]
METHODS = ["tv", "nlm", "wiener", "rl", "unet"]
METRICS = ["psnr", "ssim", "fg"]


def main():
    s = json.load(open("results/bench_summary.json"))
    targets = {
        "psnr": [30, 32, 34, 35, 36, 37, 38],
        "ssim": [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96],
        "fg": [22, 24, 26, 28, 30, 32],
    }
    out = {}
    for metric, ts in targets.items():
        mi = METRICS.index(metric)
        print(f"\n{metric}: smallest dose with >=90% FOV success")
        print("target  " + "  ".join(f"{m:>6s}" for m in METHODS) +
              "   unet/tv")
        out[metric] = []
        for T in ts:
            doses = {}
            for m in METHODS:
                arr = {d: np.asarray(s["fov"][m][str(d)])[:, mi]
                       for d in DOSES}
                hit = None
                for d in DOSES:
                    if float((arr[d] >= T).mean()) >= 0.9:
                        hit = d
                        break
                doses[m] = hit
            tv, un = doses["tv"], doses["unet"]
            ratio = (tv / un) if (tv and un) else float("nan")
            print(f"{T:6.2f}  " + "  ".join(
                f"{doses[m] if doses[m] else 0:6d}" for m in METHODS) +
                f"   {ratio:6.2f}")
            out[metric].append(dict(target=T, dose=doses,
                                    unet_vs_tv=ratio))
    json.dump(out, open("results/protocol_dose.json", "w"), indent=2)


if __name__ == "__main__":
    main()
