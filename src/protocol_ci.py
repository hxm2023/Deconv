"""Bootstrap CIs for the 90%-success protocol doses and photon factors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DOSES = [3, 10, 30, 100, 300]
B = json.load(open("results/blind_fair_doses.json"))
D = json.load(open("results/tvdeconv_doses.json"))

UNET = {m: {d: np.array([r[m] for r in B[str(d)]]) for d in DOSES}
        for m in ("psnr", "ssim", "fg")}
TV = {}
for m, (kd, kn) in {"psnr": ("tv_deconv", "tv_denoise"),
                    "ssim": ("tv_deconv_ssim", "tv_denoise_ssim"),
                    "fg": ("tv_deconv_fg", "tv_denoise_fg")}.items():
    TV[m] = {}
    for d in DOSES:
        rows = D[str(d)]["rows"]
        TV[m][d] = (np.array([np.mean(r[kd]) for r in rows]),
                    np.array([np.mean(r[kn]) for r in rows]))


def proto(vals, target):
    for d in DOSES:
        if float((np.asarray(vals[d]) >= target).mean()) >= 0.9:
            return d
    return None


def tv_best(m, idx):
    """Per-dose best TV method on the resampled FOVs; returns dict dose->vals."""
    out = {}
    for d in DOSES:
        a, b = TV[m][d][0][idx], TV[m][d][1][idx]
        out[d] = a if a.mean() >= b.mean() else b
    return out


def main():
    targets = [("psnr", t) for t in (30, 32, 34)] + \
              [("ssim", t) for t in (0.85, 0.90, 0.92)] + \
              [("fg", t) for t in (22, 24, 26)]
    rng = np.random.default_rng(0)
    out = []
    n = 20
    for m, t in targets:
        idx_all = np.arange(n)
        ud = proto(UNET[m], t)
        tvd = proto(tv_best(m, idx_all), t)
        row = dict(metric=m, target=t, unet_dose=ud, tv_dose=tvd,
                   factor=(None if not (ud and tvd) else tvd / ud))
        if ud and tvd:
            fs = []
            for _ in range(2000):
                idx = rng.integers(0, n, n)
                u = proto({d: UNET[m][d][idx] for d in DOSES}, t)
                v = proto(tv_best(m, idx), t)
                if u and v:
                    fs.append(v / u)
            fs = np.asarray(fs)
            row.update(n_boot=len(fs),
                       factor_median=float(np.median(fs)),
                       factor_ci=[float(np.percentile(fs, 2.5)),
                                  float(np.percentile(fs, 97.5))],
                       frac_ge3=float((fs >= 3).mean()),
                       frac_ge10=float((fs >= 10).mean()))
        out.append(row)
        print(row, flush=True)
    Path("results/protocol_ci.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
