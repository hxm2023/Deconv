"""Cross-source check on large AI4Life fields, one task per field."""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TUNED = {"bm3d": 0.3, "nlm": 0.2, "tv": 0.24}
CKPT = "checkpoints/real_unet.pt"
CH = 32


def work(task):
    import torch
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    from baselines import bm3d_anscombe, nlm, tv
    from model import UNet
    from real_common import load_pair

    torch.set_num_threads(1)
    stem, npath, gpath, n_patch = task
    n, g = load_pair(npath, gpath)
    ps = [(i, j) for i in range(0, n.shape[0] - 256 + 1, 256)
          for j in range(0, n.shape[1] - 256 + 1, 256)]
    step = max(1, len(ps) // n_patch)
    ps = ps[::step][:n_patch]
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = UNet(CH, 1)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    acc = {k: [] for k in ["noisy", "bm3d", "nlm", "tv", "unet"]}
    for i, j in ps:
        x = n[i:i + 256, j:j + 256]
        gt = g[i:i + 256, j:j + 256]
        with torch.no_grad():
            p = net(torch.as_tensor(x)[None, None]).cpu().numpy()[0, 0]
        preds = dict(noisy=x, bm3d=bm3d_anscombe(x, TUNED["bm3d"]),
                     nlm=nlm(x, TUNED["nlm"]), tv=tv(x, TUNED["tv"]),
                     unet=p)
        for k, im in preds.items():
            q = np.clip(im, 0, 1)
            acc[k].append([float(psnr(gt, q, data_range=1)),
                           float(ssim(gt, q, data_range=1))])
    out = {k: [float(np.mean([a[0] for a in v])),
               float(np.mean([a[1] for a in v]))] for k, v in acc.items()}
    return int(stem), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ckpt", default="checkpoints/real_unet.pt")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--out", default="results/real_large_fair_strong.json")
    a = ap.parse_args()
    global CKPT, CH
    CKPT, CH = a.ckpt, a.ch

    from real_common import pairs
    tasks = [(stem, npath, gpath, a.n_patch)
             for npath, gpath, stem in pairs("large")]
    print("field tasks", len(tasks), "workers", a.workers, flush=True)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(work, tasks)
    rows = [out for _, out in sorted(res)]
    for stem, out in sorted(res):
        print(stem, {k: round(v[0], 2) for k, v in out.items()}, flush=True)
    mean = {k: [float(np.mean([r[k][0] for r in rows])),
                float(np.mean([r[k][1] for r in rows]))] for k in rows[0]}
    print("means", {k: round(v[0], 2) for k, v in mean.items()}, flush=True)
    json.dump(dict(tuned=TUNED, rows=rows, ckpt=a.ckpt), open(a.out, "w"),
              indent=2)


if __name__ == "__main__":
    main()
