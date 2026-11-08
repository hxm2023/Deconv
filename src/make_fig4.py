"""Ablation and robustness figure for the mixed-dose model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
D = json.load(open("results/tvdeconv_doses.json"))
B = json.load(open("results/blind_fair_doses.json"))
MATCH = json.load(open("results/matched_ablation.json"))
SCR2 = json.load(open("results/scr2_matched.json"))
SIGD = json.load(open("results/tvdeconv_sigma.json"))
BLINDSIG = json.load(open("results/blind_sigma.json"))


def tvdec(dose):
    return float(np.mean([np.mean(r["tv_deconv"])
                          for r in D[str(dose)]["rows"]]))


def blind(dose):
    return float(np.mean([r["psnr"] for r in B[str(dose)]]))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), constrained_layout=True)
    doses = [3, 10, 30, 100]
    real = [blind(d) for d in doses]
    dec = [tvdec(d) for d in doses]
    scr_d = [3, 30, 100]
    scr = [float(np.mean(SCR2[str(d)])) for d in scr_d]
    ax = axes[0]
    ax.plot(doses, real, "o-", color="#d62728", lw=1.6, label="Mixed-dose")
    ax.plot(scr_d, scr, "s--", color="#7f7f7f", lw=1.4, label="Scrambled")
    ax.plot(doses, dec, "^-.", color="#1f77b4", lw=1.4, label="TV deconv")
    ax.set_xscale("log"); ax.set_xticks(doses)
    ax.set_xticklabels([str(d) for d in doses])
    ax.set_xlabel("Photons per pixel"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) learned structure control", loc="left", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    ds = [3, 30, 100]
    spec = [MATCH["specialist"][str(d)]["psnr"] for d in ds]
    cond = [blind(d) for d in ds]
    x = np.arange(len(ds))
    for i in range(len(ds)):
        ax.plot([i, i], [spec[i], cond[i]], color="black", lw=0.8, alpha=0.5)
    ax.plot(x, spec, "o", ms=7, color="#7f7f7f", label="Per-dose specialist")
    ax.plot(x, cond, "s", ms=7, color="#d62728", label="Mixed-dose")
    ax.set_xticks(x); ax.set_xticklabels([f"{d} ph" for d in ds])
    ax.set_ylabel("PSNR (dB)"); ax.set_ylim(30, 39.5)
    ax.set_title("(b) one model vs specialists", loc="left", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    ax = axes[2]
    sig = [1.5, 2.0, 2.5]
    u = [float(np.mean(BLINDSIG[str(s)])) for s in sig]
    t = [float(SIGD[str(s)][0]) for s in sig]
    x = np.arange(len(sig))
    ax.plot(x, u, "o-", ms=6, color="#d62728", label="Mixed-dose")
    ax.plot(x, t, "s-", ms=6, color="#1f77b4", label="TV deconv")
    ax.set_xticks(x); ax.set_xticklabels([f"$\\sigma$={s:g}" for s in sig])
    ax.set_ylabel("PSNR at 30 photons (dB)"); ax.set_ylim(32, 38)
    ax.set_title("(c) PSF robustness", loc="left", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.savefig(OUT / "fig4_ablations.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig4_ablations.png", dpi=300, bbox_inches="tight")
    print("wrote figures/fig4_ablations.pdf")


if __name__ == "__main__":
    main()
