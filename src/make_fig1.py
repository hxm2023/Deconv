"""Qualitative reconstruction figure: GT, noisy, TV, NLM, U-Net, error, zoom."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.2, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.linewidth": 0.4,
})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import nlm, tv_deconv
from common import dataset_paths, degrade, load_image, psnr, split_paths, ssim
from model import UNet

OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def unet_predict(net, y, dev):
    x = torch.as_tensor(y, dtype=torch.float32)[None, None].to(dev)
    with torch.no_grad():
        return net(x).cpu().numpy()[0, 0]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_blind_strong.pt", map_location=dev,
                    weights_only=False)
    ch = ck.get("args", {}).get("ch", 32)
    net = UNet(ch, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    fovs = [load_image(p, scale) for p in sp["test"]]
    doses = [3.0, 10.0, 30.0]
    lam = {3: 0.2, 10: 0.08, 30: 0.04}
    nlm_h = {3: 4.0, 10: 0.3, 30: 0.3}
    k = 0
    img = fovs[k]
    gt = img[200:392, 300:492]
    print("fixed test FOV", k, flush=True)

    nrow, ncol = 3, 6
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = GridSpec(nrow, ncol, figure=fig,
                  left=0.055, right=0.992, top=0.89, bottom=0.17,
                  wspace=0.055, hspace=0.055)
    titles = ["Ground truth", "Noisy", "TV deconv.", "NLM (Ans.)", "U-Net",
              "|U-Net - GT|"]
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 0.2))
    sm.set_array([])
    err_pos = None

    for r, d in enumerate(doses):
        rng = np.random.default_rng(7)
        y = degrade(gt, 2.0, d, rng)
        p_tv = tv_deconv(y, 2.0, lam[int(d)], rho=0.3, iters=30, pad=32)
        p_nlm = nlm(y, nlm_h[int(d)])
        p_un = unet_predict(net, y, dev)
        err = np.abs(np.clip(p_un, 0, 1) - gt)
        panels = [(gt, None), (y, psnr(gt, y)), (p_tv, psnr(gt, p_tv)),
                  (p_nlm, psnr(gt, p_nlm)), (p_un, psnr(gt, p_un))]
        for c, (im, val) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("0.75"); s.set_linewidth(0.35)
            if r == 0:
                ax.set_title(titles[c], fontsize=7.0, pad=2.5)
            if val is not None:
                s = ssim(gt, im)
                ax.text(0.03, 0.04, f"{val:.1f} dB\nSSIM {s:.2f}",
                        transform=ax.transAxes, va="bottom", ha="left",
                        fontsize=5.4, color="white", linespacing=1.15,
                        bbox=dict(fc="black", alpha=0.45, ec="none",
                                  pad=1.2))
            if r == 0 and c == 0:
                ax.plot([8, 28], [182, 182], color="white", lw=1.4,
                        solid_capstyle="butt")
                ax.text(12, 166, "20 px", color="white", ha="left",
                        va="bottom", fontsize=5.6)
            if r == 0 and c in (2, 4):
                roi = (slice(96, 156), slice(16, 76))
                ax.add_patch(Rectangle((16, 96), 60, 60, fill=False,
                                       edgecolor="#FFD166", lw=0.8))
                iax = ax.inset_axes([0.56, 0.54, 0.42, 0.42])
                iax.imshow(np.clip(im[roi], 0, 1), cmap="gray",
                           vmin=0, vmax=1)
                iax.set_xticks([]); iax.set_yticks([])
                for s in iax.spines.values():
                    s.set_color("#FFD166"); s.set_linewidth(0.8)
        ax = fig.add_subplot(gs[r, 5])
        ax.imshow(err, cmap="magma", vmin=0, vmax=0.2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.75"); s.set_linewidth(0.35)
        if r == 2:
            err_pos = ax.get_position()
        if r == 0:
            ax.set_title(titles[5], fontsize=7.0, pad=2.5)
        fig.text(0.012, 0.73 - r * 0.265, f"$P={int(d)}$", rotation=90,
                 va="center", ha="center", fontsize=7.5)

    # shared colorbar under the error column
    pos = err_pos
    cax = fig.add_axes([pos.x0, pos.y0 - 0.045, pos.width, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0.0, 0.1, 0.2])
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=2)
    cb.set_label("absolute error", fontsize=5.8, labelpad=1)
    cb.outline.set_linewidth(0.35)

    fig.savefig(OUT / "fig1_qualitative.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig1_qualitative.png", dpi=300, bbox_inches="tight")
    print("wrote figures/fig1_qualitative.pdf", flush=True)


if __name__ == "__main__":
    main()
