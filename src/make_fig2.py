"""Dose-response, gain, and validation-selected protocol figure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
DOSES = [3, 10, 30, 100, 300]
METRICS = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"),
           ("fg", "Foreground PSNR (dB)")]
COL = {"unet": "#C44536", "tv": "#2E6F95", "den": "#7A8B99"}
D = json.load(open("results/tvdeconv_doses.json"))
B = json.load(open("results/blind_fair_doses.json"))
try:
    PVC = {(r["metric"], float(r["target"])): r
           for r in json.load(open("results/protocol_val.json"))["targets"]}
except FileNotFoundError:
    PVC = None
try:
    PCI = {(r["metric"], float(r["target"])): r
           for r in json.load(open("results/protocol_ci.json"))}
except FileNotFoundError:
    PCI = {}


def fov(dose, key):
    return np.array([np.mean(r[key]) for r in D[str(dose)]["rows"]])


def bfov(dose, metric):
    return np.array([r[metric] for r in B[str(dose)]])


def mean_ci(a):
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def main():
    fig = plt.figure(figsize=(7.2, 5.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.45],
                          hspace=0.48, wspace=0.38,
                          left=0.075, right=0.985, top=0.90, bottom=0.10)
    handles, labels = None, None
    for j, (metric, ylab) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j])
        for key, label, color, marker in [
                ("unet", "U-Net", COL["unet"], "o"),
                ("tv_deconv", "TV deconvolution", COL["tv"], "s"),
                ("tv_denoise", "TV denoising", COL["den"], "^")]:
            if key == "unet":
                vals = [bfov(d, metric) for d in DOSES]
            else:
                k = key if metric == "psnr" else key + "_" + metric
                vals = [fov(d, k) for d in DOSES]
            mu = np.array([v.mean() for v in vals])
            ci = np.array([1.96 * v.std(ddof=1) / np.sqrt(len(v))
                           for v in vals])
            ax.fill_between(DOSES, mu - ci, mu + ci, color=color,
                            alpha=0.12, lw=0)
            ax.plot(DOSES, mu, marker=marker, ms=4.2, lw=1.5, color=color,
                    label=label, markeredgewidth=0.4,
                    markeredgecolor="white")
        ax.set_xscale("log")
        ax.set_xticks(DOSES)
        ax.set_xticklabels([str(d) for d in DOSES])
        ax.minorticks_off()
        ax.set_xlabel("Dose $P$ (full-scale units)", labelpad=1)
        ax.set_ylabel(ylab, labelpad=1)
        ax.grid(alpha=0.20, lw=0.45)
        ax.set_title(f"({chr(97+j)}) {ylab}", loc="left", fontsize=8,
                     fontweight="bold")
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()

    # (d) paired gain
    ax = fig.add_subplot(gs[0, 3])
    mu, ci = [], []
    for d in DOSES:
        diff = bfov(d, "psnr") - fov(d, "tv_deconv")
        m, c = mean_ci(diff)
        mu.append(m); ci.append(c)
    mu = np.array(mu); ci = np.array(ci)
    ax.axhline(0, color="0.45", lw=0.7, ls="--")
    ax.errorbar(DOSES, mu, yerr=ci, color=COL["unet"], marker="o", ms=4.2,
                lw=1.5, capsize=2, elinewidth=0.8, markeredgecolor="white",
                markeredgewidth=0.4)
    ax.set_ylim(0.55, 1.80)
    ax.set_xlim(2.4, 420)
    for i, (d, m, c) in enumerate(zip(DOSES, mu, ci)):
        ha = "left" if i == 0 else ("right" if i == len(DOSES) - 1 else "center")
        ax.text(d, m + c + 0.035, f"{m:.2f}", ha=ha, va="bottom",
                fontsize=5.8, color=COL["unet"])
    ax.set_xscale("log")
    ax.set_xticks(DOSES)
    ax.set_xticklabels([str(d) for d in DOSES])
    ax.minorticks_off()
    ax.set_xlabel("Dose $P$ (full-scale units)", labelpad=1)
    ax.set_ylabel("U-Net $-$ TV (dB)", labelpad=1)
    ax.set_title("(d) paired PSNR gain", loc="left", fontsize=8,
                 fontweight="bold")
    ax.grid(alpha=0.20, lw=0.45)
    for s in ax.spines.values():
        s.set_linewidth(0.5)

    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965), fontsize=6.5,
               handlelength=1.6, columnspacing=1.6)

    # (e) protocol dot plot
    ax = fig.add_subplot(gs[1, :])
    items = [("psnr", 30, "PSNR 30 dB"), ("psnr", 32, "PSNR 32 dB"),
             ("psnr", 34, "PSNR 34 dB"),
             ("ssim", 0.85, "SSIM 0.85"), ("ssim", 0.90, "SSIM 0.90"),
             ("ssim", 0.92, "SSIM 0.92"),
             ("fg", 22, "FG 22 dB"), ("fg", 24, "FG 24 dB"),
             ("fg", 26, "FG 26 dB")]
    ypos = np.arange(len(items))[::-1]
    for yi, (metric, T, lab) in zip(ypos, items):
        if PVC is None:
            continue
        r = PVC[(metric, float(T))]
        tv, un = r["tv_dose"], r["unet_dose"]
        ok = ((r["tv_test_success"] or 0) >= 0.9 and
              (r["unet_test_success"] or 0) >= 0.9)
        ax.plot([tv, un], [yi, yi], color="0.65", lw=1.0, zorder=1)
        ax.scatter([tv], [yi], s=34, color=COL["tv"], edgecolor="white",
                   linewidth=0.6, zorder=3)
        ax.scatter([un], [yi], s=16, color=COL["unet"], edgecolor="white",
                   linewidth=0.4, zorder=4)
        if ok and r["factor"] and r["factor"] > 1.01:
            xm = np.sqrt(tv * un)
            ax.text(xm, yi + 0.22, f"{r['factor']:.1f}$\\times$",
                    ha="center", va="bottom", fontsize=6.0,
                    color="0.15")
        elif ok:
            ax.text(np.sqrt(tv * un), yi + 0.22, "tie", ha="center",
                    va="bottom", fontsize=5.8, color="0.25")
        else:
            bad_un = (r["unet_test_success"] or 0) < 0.9
            bad_x = un if bad_un else tv
            who = "U-Net" if bad_un else "TV"
            ax.text(bad_x * 1.12, yi + 0.20, f"{who} 80%", ha="left",
                    va="bottom", fontsize=5.2, color=COL["unet"])
        su = 100 * (r["unet_test_success"] or 0)
        st = 100 * (r["tv_test_success"] or 0)
        ax.text(300, yi, f"{su:.0f}/{st:.0f}", ha="center", va="center",
                fontsize=5.7, color="0.20")
        p3 = PCI.get((metric, float(T)), {}).get("frac_ge3")
        if p3 is not None:
            ax.text(1800, yi, f"{p3:.2f}", ha="center", va="center",
                    fontsize=5.7, color="0.20")
    ax.set_xscale("log")
    ax.set_xlim(2.2, 5000)
    ax.set_ylim(-0.5, 9.3)
    ax.text(300, 8.45, "success %\nU-Net/TV", ha="center", va="bottom",
            fontsize=5.2, color="0.25", linespacing=0.95)
    ax.text(1800, 8.45, "P(save\n$\\geq3\\times$)", ha="center", va="bottom",
            fontsize=5.2, color="0.25", linespacing=0.95)
    ax.axvline(180, color="0.88", lw=0.5, ls=":")
    ax.axvline(900, color="0.88", lw=0.5, ls=":")
    ax.set_xticks(DOSES[:-1])
    ax.set_xticklabels([str(d) for d in DOSES[:-1]])
    ax.minorticks_off()
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, _, lab in items], fontsize=6.8)
    ax.set_xlabel("Validation-selected dose $P$ (full-scale units)",
                  labelpad=1)
    ax.set_title("(e) dose at 90% field-of-view success",
                 loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="x", alpha=0.18, lw=0.45)
    for s in ax.spines.values():
        s.set_linewidth(0.5)

    fig.savefig(OUT / "fig2_dose_response.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig2_dose_response.png", dpi=300, bbox_inches="tight")
    print("wrote figures/fig2_dose_response.pdf", flush=True)


if __name__ == "__main__":
    main()
