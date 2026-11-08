"""FMD second-source figure: three microscope examples plus paired scatter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import bm3d_anscombe
from model import UNet
from real_common import norm99

plt.rcParams.update({"font.size": 7, "axes.titlesize": 6.8,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42})
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
FIELDS = json.load(open("results/fmd_test_fields.json"))
ROWS = json.load(open("results/fmd_test.json"))["rows"]
SIGMA = 0.10

net = UNet(32, 1)
ck = torch.load("checkpoints/fmd_unet.pt", map_location="cpu",
                weights_only=False)
net.load_state_dict(ck["state_dict"])
net.eval()

fig = plt.figure(figsize=(7.2, 4.35))
gs = GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1.22],
              wspace=0.30, hspace=0.30, left=0.01, right=0.985,
              bottom=0.04, top=0.92)

for r, idx in enumerate([0, 12, 24]):
    rec = FIELDS[idx]
    n = norm99(io.imread("datasets/fmd/" + rec["noisy"]).astype(np.float32))
    g = norm99(io.imread("datasets/fmd/" + rec["gt"]).astype(np.float32))
    with torch.no_grad():
        u = net(torch.as_tensor(n)[None, None]).numpy()[0, 0]
    b = bm3d_anscombe(n, SIGMA)
    pn = psnr(g, np.clip(n, 0, 1), data_range=1)
    pb = psnr(g, np.clip(b, 0, 1), data_range=1)
    pu = psnr(g, np.clip(u, 0, 1), data_range=1)
    panels = [(n, f"noisy  {pn:.1f} dB"), (g, "clean reference"),
              (b, f"BM3D  {pb:.1f} dB"), (u, f"U-Net  {pu:.1f} dB")]
    for c, (im, title) in enumerate(panels):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(np.clip(im, 0, 1), cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        if c == 1:
            ax.plot([8, 28], [248, 248], color="white", lw=1.4)
            ax.text(8, 236, "20 px", color="white", va="bottom", ha="left",
                    fontsize=5.5)

ax = fig.add_subplot(gs[:, 4])
x = np.array([r["bm3d"][0] for r in ROWS])
y = np.array([r["unet"][0] for r in ROWS])
ax.scatter(x, y, s=10, c="#1f77b4", alpha=0.85, linewidths=0)
lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
ax.plot(lim, lim, "--", color="0.45", lw=0.7)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("BM3D PSNR (dB)", labelpad=1)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_ylabel("U-Net PSNR (dB)", labelpad=2)
ax.set_title(f"held-out FMD fields (n = {len(ROWS)})", pad=3)
d = y - x
ax.text(0.04, 0.96, f"U-Net $-$ BM3D\nmean {d.mean():+.2f} dB\n"
        f"wins {(d > 0).sum()}/{len(d)}",
        transform=ax.transAxes, va="top", fontsize=6.8)
ax.tick_params(width=0.4, length=2)
for s in ax.spines.values():
    s.set_linewidth(0.5)

fig.savefig(OUT / "fig5_fmd.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig5_fmd.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig5_fmd.pdf")
