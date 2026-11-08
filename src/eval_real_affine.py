"""Affine-matched classical baseline control on real AI4Life pairs.

The supervised U-Net is trained to predict an independently normalized clean
image, so it can learn a global gain/offset that BM3D/NLM/TV cannot. Here we
fit one affine map per classical method on validation pairs and apply it to
the held-out test predictions, then compare with the U-Net.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TUNED = {"bm3d": 0.3, "nlm": 0.2, "tv": 0.24}
CKPT = "checkpoints/real_unet.pt"
CH = 32


def work(item):
    import torch
    from baselines import bm3d_anscombe, nlm, tv
    from model import UNet
    from real_common import load_pair

    torch.set_num_threads(1)
    npath, gpath, stem = item
    n, g = load_pair(npath, gpath)
    dev = "cpu"
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    net = UNet(CH, 1).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        u = net(torch.as_tensor(n)[None, None]).cpu().numpy()[0, 0]
    return str(stem), n, g, dict(
        bm3d=bm3d_anscombe(n, TUNED["bm3d"]),
        nlm=nlm(n, TUNED["nlm"]),
        tv=tv(n, TUNED["tv"]),
        unet=u,
    )


def fit_affine(preds, gts):
    x = np.concatenate([p.ravel() for p in preds])
    y = np.concatenate([g.ravel() for g in gts])
    A = np.c_[x, np.ones_like(x)]
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def metrics(pred, gt):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    p = np.clip(pred, 0, 1)
    return [float(psnr(gt, p, data_range=1)),
            float(ssim(gt, p, data_range=1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt", default="checkpoints/real_unet.pt")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--out", default="results/real_affine_test40.json")
    a = ap.parse_args()
    global CKPT, CH
    CKPT, CH = a.ckpt, a.ch
    from real_common import split_small
    _, val, test = split_small(0)
    val = val[:a.n_val]
    test = test[:a.n_test]
    items = val + test
    print("tasks", len(items), flush=True)
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(work, items)
    print("elapsed", round(time.time() - t0, 1), flush=True)
    val_res, test_res = res[:len(val)], res[len(val):]

    methods = ["bm3d", "nlm", "tv"]
    affine = {}
    for m in methods:
        affine[m] = fit_affine([r[3][m] for r in val_res],
                               [r[2] for r in val_res])
        print("affine", m, affine[m], flush=True)

    rows = []
    for stem, n, g, preds in test_res:
        row = dict(stem=stem)
        for m in methods:
            av, bv = affine[m]
            row[m] = metrics(av * preds[m] + bv, g)
            row[m + "_raw"] = metrics(preds[m], g)
        row["unet"] = metrics(preds["unet"], g)
        rows.append(row)
    summary = {}
    for k in ["unet", "bm3d", "bm3d_raw", "nlm", "nlm_raw", "tv", "tv_raw"]:
        summary[k] = [float(np.mean([r[k][0] for r in rows])),
                      float(np.mean([r[k][1] for r in rows]))]
        print(k, "PSNR", round(summary[k][0], 2), "SSIM",
              round(summary[k][1], 3), flush=True)
    from scipy.stats import wilcoxon
    d = np.array([r["unet"][0] - r["bm3d"][0] for r in rows])
    print("unet-affineBM3D", round(float(d.mean()), 2),
          "wins", int((d > 0).sum()), "p", float(wilcoxon(d).pvalue),
          flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tuned=TUNED, affine=affine, rows=rows,
                   summary=summary), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
