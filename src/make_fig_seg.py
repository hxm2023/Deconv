"""Downstream nuclei segmentation: Dice versus dose and example masks."""
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
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv, tv_deconv
from common import bbbc039_official_split, degrade, patches
from seg_common import dice, load_pair
from seg_model import MultiTaskUNet

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5})
OUT = Path("figures"); OUT.mkdir(exist_ok=True)


def otsu(img):
    try:
        return img > threshold_otsu(img)
    except Exception:
        return img > float(img.mean())


def matched_dose(curve, target):
    xs = np.log10(list(curve.keys())); ys = list(curve.values())
    if target <= ys[0]:
        return float(10 ** xs[0])
    for k in range(1, len(ys)):
        if ys[k] >= target:
            x0, x1 = xs[k - 1], xs[k]
            y0, y1 = ys[k - 1], ys[k]
            return float(10 ** (x0 + (target - y0) / (y1 - y0) * (x1 - x0)))
    return float(10 ** xs[-1])


def main():
    d = json.load(open("results/seg_doses_full.json"))
    doses = sorted(float(k) for k in d)
    curves = {m: {p: d[str(float(p))][m] for p in doses}
              for m in ["noisy", "tvden", "tvdec", "unet"]}
    print("matched doses")
    for P in [1, 3, 10]:
        t = curves["unet"][P]
        print(f"  U-Net P={P} Dice={t:.3f} -> TVdec {matched_dose(curves['tvdec'], t):.2f}, "
              f"TVden {matched_dose(curves['tvden'], t):.2f}", flush=True)

    fig = plt.figure(figsize=(7.2, 3.1), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    style = [("noisy", "Noisy", "#000000", ":"), ("tvden", "TV denoise", "#7f7f7f", "--"),
             ("tvdec", "TV deconv", "#1f77b4", "-."),
             ("unet", "U-Net", "#d62728", "-")]
    for key, lab, col, ls in style:
        y = [curves[key][p] for p in doses]
        ax.plot(doses, y, ls, marker="o", ms=3.5, lw=1.4, color=col, label=lab)
    ax.set_xscale("log"); ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_xlabel("Photons per pixel (full scale)")
    ax.set_ylabel("Nuclei Dice")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("(a) downstream segmentation", loc="left", fontsize=8)

    # example masks on a validation FOV at P=1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/seg_rest_full.pt", map_location=dev,
                    weights_only=False)
    net = MultiTaskUNet(32, 1, restore=True, segment=False).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    sp = bbbc039_official_split()
    img, mask = load_pair(sp["val"][0])
    gt = img[200:392, 300:492]
    mt = mask[200:392, 300:492]
    rng = np.random.default_rng(2026)
    y = degrade(gt, 2.0, 1.0, rng)
    with torch.no_grad():
        xt = torch.as_tensor(y, dtype=torch.float32)[None, None].to(dev)
        rest, _ = net(xt)
    pun = rest.cpu().numpy()[0, 0]
    pdec = tv_deconv(y, 2, 0.04, iters=30)
    pden = tv(y, 0.04)
    panels = [(mt, "Ground truth", None),
              (otsu(y), "Noisy + Otsu", dice(mt, otsu(y))),
              (otsu(pdec), "TV deconv + Otsu", dice(mt, otsu(pdec))),
              (otsu(pun), "U-Net + Otsu", dice(mt, otsu(pun)))]
    # 2x2 grid of masks
    sub = gs[0, 1].subgridspec(2, 2, hspace=0.18, wspace=0.08)
    for k, (im, name, dv) in enumerate(panels):
        a = fig.add_subplot(sub[k // 2, k % 2])
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(name + ("" if dv is None else f"\nDice {dv:.3f}"),
                    fontsize=6.2)
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(OUT / "fig4_segmentation.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig4_segmentation.png", dpi=300, bbox_inches="tight")
    print("wrote figures/fig4_segmentation.pdf")


if __name__ == "__main__":
    main()
