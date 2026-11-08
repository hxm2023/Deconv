"""Cross-dataset gains and ensemble-uncertainty calibration figure."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5})
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_eval import ensemble_predict, load_nets
from common import dataset_paths, degrade, load_image, patches, psnr, split_paths

OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def roc_auc(scores, labels):
    order = np.argsort(-scores)
    lab = labels[order].astype(float)
    tpr = np.r_[0, np.cumsum(lab) / max(lab.sum(), 1)]
    fpr = np.r_[0, np.cumsum(1 - lab) / max((1 - lab).sum(), 1)]
    return float(np.trapezoid(tpr, fpr)), tpr, fpr


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n)]
    return 1.96 * float(np.std(v))


def compute_uncertainty():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = load_nets(["checkpoints/bbbc014_dcond.pt",
                      "checkpoints/bbbc014_dcond_s1.pt",
                      "checkpoints/bbbc014_dcond_s2.pt"], dev)
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    test = [load_image(p, scale) for p in sp["test"]]
    doses = [3.0, 10.0, 30.0]
    rows = []
    for fi, img in enumerate(test):
        ps = list(patches(img, 128))
        step = max(1, len(ps) // 10)
        ps = ps[::step][:10]
        gts = [gt for _, _, gt in ps]
        row = dict(fov=fi)
        for d in doses:
            rng = np.random.default_rng(51)
            ys = [degrade(gt, 2.0, d, rng) for gt in gts]
            mean, var = ensemble_predict(nets, ys, d, dev)
            mse = np.mean([(np.clip(p, 0, 1) - gt) ** 2
                           for gt, p in zip(gts, mean)])
            row[str(int(d))] = dict(
                psnr=float(np.mean([psnr(gt, p) for gt, p in zip(gts, mean)])),
                unc=float(np.mean(np.sqrt(np.clip(var, 0, None)))),
                err=float(np.sqrt(mse)))
        rows.append(row)
    json.dump(rows, open("results/uncertainty_fov.json", "w"), indent=2)
    return rows, test, nets, dev


def main():
    fov_rows, test, nets, dev = compute_uncertainty()

    ext = json.load(open("results/tvdeconv_external30.json"))
    maind = json.load(open("results/tvdeconv_doses.json"))
    blind = json.load(open("results/blind_fair_doses.json"))
    blind_ext = json.load(open("results/blind_external30.json"))
    sets = [("main", "BBBC014"), ("bbbc015", "BBBC015"),
            ("bbbc016", "BBBC016"), ("bbbc018", "BBBC018")]
    deltas = {}
    d_main = np.array([r["psnr"] for r in blind["30"]]) - np.array(
        [np.mean(r["tv_deconv"]) for r in maind["30"]["rows"]])
    deltas["BBBC014"] = (float(d_main.mean()), boot_ci(d_main))
    try:
        tuned = json.load(open("results/external_tuned.json"))
    except FileNotFoundError:
        tuned = None
    for key, name in sets[1:]:
        if tuned:
            d = np.array(tuned[key]["delta"])
        else:
            d = np.array(blind_ext[key]) - np.array(
                [r["tvdec"] for r in ext[key]])
        deltas[name] = (float(d.mean()), boot_ci(d))

    # example maps at 3 photons
    vpaths, vscale = dataset_paths("bbbc014")
    vsp = split_paths(vpaths, 0, 20, 20)
    val = [load_image(p, vscale) for p in vsp["val"]]
    img = val[0]
    gt = img[250:442, 350:542]
    rng = np.random.default_rng(51)
    y = degrade(gt, 2.0, 3.0, rng)
    mean, var = ensemble_predict(nets, [y], 3.0, dev)
    mean, var = mean[0], var[0]
    err = np.abs(np.clip(mean, 0, 1) - gt)

    fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.85, 0.95, 0.85])

    # (a) cross-dataset gains
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(sets))
    means = [deltas[n][0] for _, n in sets]
    errs = [deltas[n][1] for _, n in sets]
    ax.bar(x, means, 0.58, yerr=errs, capsize=2, edgecolor="white",
           linewidth=0.4, color=["#C44536", "#2E6F95", "#2E6F95", "#2E6F95"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in sets], fontsize=8)
    ax.set_ylabel("U-Net PSNR gain (dB)")
    ax.set_title("(a) cross-dataset gain at $P=30$", loc="left", fontsize=9,
                 fontweight="bold")
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(i, m + e + 0.04, f"{m:+.2f}", ha="center", va="bottom",
                fontsize=6.2)
    ax.set_ylim(0, max(np.array(means) + np.array(errs)) * 1.28)
    ax.grid(axis="y", alpha=0.20, lw=0.45)

    # (b) ROC
    ax = fig.add_subplot(gs[0, 2:])
    aucs = {}
    uci = json.load(open("results/uncertainty_ci.json"))
    for d, col in [(3, "#C44536"), (10, "#E09F3E")]:
        scores = np.array([r[str(d)]["unc"] for r in fov_rows])
        labels = np.array([r[str(d)]["psnr"] < 34 for r in fov_rows])
        auc, tpr, fpr = roc_auc(scores, labels)
        aucs[d] = auc
        lo, hi = uci[str(d)]["ci"]
        ax.plot(fpr, tpr, lw=1.6, color=col,
                label=f"{d} photons (AUC {auc:.2f} [{lo:.2f},{hi:.2f}], "
                      f"n={int(labels.sum())})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(b) failure detection (PSNR < 34 dB)", loc="left", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    # maps
    maps = [(gt, "Ground truth", "gray", dict(vmin=0, vmax=1)),
            (y, "Noisy, $P=3$", "gray", dict(vmin=0, vmax=1)),
            (np.clip(mean, 0, 1), "Ensemble mean", "gray", dict(vmin=0, vmax=1)),
            (np.sqrt(np.clip(var, 0, None)), "Uncertainty", "magma",
             dict(vmin=0, vmax=float(np.quantile(np.sqrt(var), 0.99)))),
            (err, "|Error|", "inferno", dict(vmin=0, vmax=0.15))]
    for c, (im, name, cmap, kw) in enumerate(maps[:4]):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(im, cmap=cmap, **kw)
        ax.set_box_aspect(1)
        ax.set_title(("(c) " if c == 0 else "") + name, fontsize=8,
                     fontweight="bold" if c == 0 else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        if name == "Ground truth":
            ax.plot([8, 28], [184, 184], color="white", lw=1.2)
            ax.text(8, 174, "20 px", color="white", fontsize=5)
        if name == "Ensemble mean":
            ax.text(0.04, 0.04, f"PSNR {psnr(gt, np.clip(mean, 0, 1)):.1f} dB",
                    transform=ax.transAxes, va="bottom", ha="left",
                    fontsize=5.2, color="white",
                    bbox=dict(fc="black", alpha=0.45, ec="none", pad=1.0))
        if name == "Uncertainty":
            cax = ax.inset_axes([0.06, 0.035, 0.56, 0.045])
            cb = fig.colorbar(ax.images[0], cax=cax, orientation="horizontal")
            cb.set_label("std", fontsize=4.8, labelpad=0.5)
            cb.ax.tick_params(labelsize=4.8, width=0.3, length=1.2)

    # (c) calibration
    ax = fig.add_subplot(gs[2, :2])
    u = np.array([r["3"]["unc"] for r in fov_rows])
    e = np.array([r["3"]["err"] for r in fov_rows])
    rho = spearmanr(u, e).statistic
    ax.scatter(u, e, s=22, c="#C44536", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Ensemble uncertainty (mean std)")
    ax.set_ylabel("Reconstruction RMSE")
    coef = np.polyfit(u, e, 1)
    xx = np.linspace(u.min(), u.max(), 50)
    ax.plot(xx, np.polyval(coef, xx), "--", color="0.35", lw=0.9)
    ax.set_title(f"(d) uncertainty vs error (rho={rho:.2f}, n=20)",
                 loc="left", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25, lw=0.5)

    # (d) re-acquisition trade-off
    ax = fig.add_subplot(gs[2, 2:])
    for d, col in [(3, "#C44536"), (10, "#E09F3E")]:
        scores = np.array([r[str(d)]["unc"] for r in fov_rows])
        labels = np.array([r[str(d)]["psnr"] < 34 for r in fov_rows])
        frac, rec = [], []
        for q in np.linspace(0, 1, 21):
            flag = scores >= np.quantile(scores, q)
            frac.append(float(flag.mean()))
            rec.append(float((flag & labels).sum() / max(labels.sum(), 1)))
        ax.plot(frac, rec, "o-", ms=2.6, lw=1.2, color=col,
                label=f"$P={d}$")
    ax.set_xlabel("Fraction of FOVs re-acquired")
    ax.set_ylabel("Fraction of failing FOVs captured")
    ax.set_title("(e) re-acquisition trade-off", loc="left", fontsize=9,
                 fontweight="bold")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)

    fig.savefig(OUT / "fig3_cross_uncertainty.pdf", dpi=300,
                bbox_inches="tight")
    fig.savefig(OUT / "fig3_cross_uncertainty.png", dpi=300,
                bbox_inches="tight")
    print("wrote figures/fig3_cross_uncertainty.pdf")


if __name__ == "__main__":
    main()
