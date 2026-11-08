"""Zero-shot external comparison: TV deconvolution vs dose-conditioned U-Net."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv_deconv
from common import dataset_paths, degrade, load_image, patches, psnr, save_json
from model import UNet


def unet_predict(net, ys, dose, dev):
    c = (math.log10(dose) - math.log10(3.0)) / (math.log10(300.0) - math.log10(3.0))
    out = []
    with torch.no_grad():
        for q in range(0, len(ys), 64):
            x = torch.as_tensor(np.asarray(ys[q:q + 64]),
                                dtype=torch.float32)[:, None]
            ch = torch.full((len(x), 1, x.shape[2], x.shape[3]), c)
            out.append(net(torch.cat([x, ch], 1).to(dev)).cpu().numpy()[:, 0])
    return np.concatenate(out, 0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_dcond.pt", map_location=dev,
                    weights_only=False)
    net = UNet(32, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    out = {}
    for key in ["bbbc015", "bbbc016", "bbbc018"]:
        ep, scale = dataset_paths(key)
        ep = ep[::max(1, len(ep) // 20)][:20]
        fovs = [load_image(p, scale) for p in ep]
        rows = []
        for img in fovs:
            ps = list(patches(img, 128))
            step = max(1, len(ps) // 10)
            gts = [g for _, _, g in ps[::step][:10]]
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, 30.0, rng) for g in gts]
            pdec = [tv_deconv(y, 2.0, 0.04, rho=0.3, iters=30, pad=32) for y in ys]
            pun = unet_predict(net, ys, 30.0, dev)
            rows.append(dict(
                tvdec=float(np.mean([psnr(g, p) for g, p in zip(gts, pdec)])),
                unet=float(np.mean([psnr(g, p) for g, p in zip(gts, pun)]))))
        out[key] = rows
        print(key, "TVdec", round(np.mean([r["tvdec"] for r in rows]), 2),
              "UNet", round(np.mean([r["unet"] for r in rows]), 2), flush=True)
    save_json("results/tvdeconv_external30.json", out)


if __name__ == "__main__":
    main()
