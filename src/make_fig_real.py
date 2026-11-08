"""Real-data validation figure: held-out pairs, cross-source failure, scatter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import bm3d_anscombe
from model import UNet
from real_common import load_pair, norm99, split_small

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 6.8, "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3, "pdf.fonttype": 42, "axes.linewidth": 0.5,
})
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--affine",
                default="results/real_affine_test_all_strong.json")
ap.add_argument("--ckpt", default="checkpoints/real_unet_strong.pt")
ap.add_argument("--ch", type=int, default=48)
args = ap.parse_args()

D = json.load(open("results/real_small_test_all.json"))
rows = D["rows"]
tuned = D["tuned"]
A = json.load(open(args.affine))
arows = A["rows"]
aff = A["affine"]["bm3d"]
_, _, test = split_small(0)

dev = "cpu"
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
net = UNet(args.ch, 1).to(dev)
net.load_state_dict(ck["state_dict"])
net.eval()

fig = plt.figure(figsize=(7.2, 4.55))
gs = GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1.35],
              wspace=0.10, hspace=0.16, left=0.045, right=0.985,
              bottom=0.10, top=0.91)
col_titles = ["Noisy", "Reference", "BM3D", "U-Net"]

# two representative held-out pairs: 25th and 75th percentile of U-Net PSNR
order = np.argsort([r["unet"][0] for r in arows])
idxs = [int(order[60]), int(order[182])]
for r, idx in enumerate(idxs):
    n, g = load_pair(test[idx][0], test[idx][1])
    with torch.no_grad():
        u = net(torch.as_tensor(n)[None, None]).numpy()[0, 0]
    b = bm3d_anscombe(n, tuned["bm3d"])
    b = np.clip(aff[0] * b + aff[1], 0, 1)
    pn = rows[idx]["noisy"][0]
    pb = arows[idx]["bm3d"][0]
    pu = arows[idx]["unet"][0]
    vals = [pn, None, pb, pu]
    panels = [n, g, b, u]
    for c, (im, val) in enumerate(zip(panels, vals)):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(np.clip(im, 0, 1), cmap="magma", vmin=0, vmax=1)
        title = col_titles[c]
        if c == 0:
            title = f"({chr(97+r)}) " + title
        if val is not None:
            title += f"  {val:.1f} dB"
        ax.set_title(title, pad=2.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.75"); s.set_linewidth(0.35)
        if c == 1:
            ax.plot([8, 28], [248, 248], color="white", lw=1.4,
                    solid_capstyle="butt")
            ax.text(8, 236, "20 px", color="white", va="bottom", ha="left",
                    fontsize=5.4)

# cross-source failure: fixed BBBC006 field 1000, fixed crop
ln = norm99(__import__("skimage").io.imread(
    "datasets/real_nuclei/large_images/noisy/1000.tif").astype(np.float32))
lg = norm99(__import__("skimage").io.imread(
    "datasets/real_nuclei/large_images/gt/1000.tif").astype(np.float32))
sl = (slice(384, 640), slice(384, 640))
ln, lg = ln[sl], lg[sl]
with torch.no_grad():
    lu = net(torch.as_tensor(ln)[None, None]).numpy()[0, 0]
lb = bm3d_anscombe(ln, tuned["bm3d"])
pn = psnr(lg, np.clip(ln, 0, 1), data_range=1)
pb = psnr(lg, np.clip(lb, 0, 1), data_range=1)
pu = psnr(lg, np.clip(lu, 0, 1), data_range=1)
vals = [pn, None, pb, pu]
block_pos = []
im_artist = None
for c, (im, val) in enumerate(zip([ln, lg, lb, lu], vals)):
    ax = fig.add_subplot(gs[2, c])
    im_artist = ax.imshow(np.clip(im, 0, 1), cmap="magma", vmin=0, vmax=1)
    block_pos.append(ax.get_position())
    title = col_titles[c].replace("BM3D", "BM3D (raw)")
    if c == 0:
        title = "(c) " + title
    if val is not None:
        title += f"  {val:.1f} dB"
    ax.set_title(title, pad=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.75"); s.set_linewidth(0.35)
    if c == 1:
        ax.plot([8, 28], [248, 248], color="white", lw=1.4,
                solid_capstyle="butt")
        ax.text(8, 236, "20 px", color="white", va="bottom", ha="left",
                fontsize=5.4)
    if c == 2:
        ax.text(0.96, 0.04, "raw", transform=ax.transAxes, va="bottom",
                ha="right", fontsize=5.2, color="white",
                bbox=dict(fc="black", alpha=0.45, ec="none", pad=1.0))

# one shared intensity colorbar for the image panels
p0, p3 = block_pos[0], block_pos[-1]
cax = fig.add_axes([p0.x0, p0.y0 - 0.045, p3.x1 - p0.x0, 0.016])
cb = fig.colorbar(im_artist, cax=cax, orientation="horizontal")
cb.set_label("normalized intensity", fontsize=5.4, labelpad=0.8)
cb.ax.tick_params(labelsize=5.0, width=0.3, length=1.2)
cb.outline.set_linewidth(0.35)

# scatter over all held-out pairs
ax = fig.add_subplot(gs[:, 4])
x = np.array([r["bm3d"][0] for r in arows])
y = np.array([r["unet"][0] for r in arows])
d = y - x
ax.scatter(x, y, c="#C44536", s=9, alpha=0.75, linewidths=0)
lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
ax.plot(lim, lim, "--", color="0.45", lw=0.7)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("BM3D PSNR (dB)", labelpad=1)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_ylabel("U-Net PSNR (dB)", labelpad=2)
ax.set_title(f"(d) {len(d)} held-out pairs", pad=3, fontweight="bold")
ax.text(0.96, 0.04,
        f"mean {d.mean():+.2f} dB\nwins {(d > 0).sum()}/{len(d)}\n"
        "above diagonal: U-Net better",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=6.3)
ax.tick_params(width=0.4, length=2)
for s in ax.spines.values():
    s.set_linewidth(0.5)

fig.savefig(OUT / "fig4_real.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig4_real.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig4_real.pdf")
