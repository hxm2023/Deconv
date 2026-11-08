"""Cross-source check on the 20 large AI4Life fields (parallel patches)."""
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
    stem, i, j, npath, gpath = task
    n, g = load_pair(npath, gpath)
    x = n[i:i + 256, j:j + 256]
    gt = g[i:i + 256, j:j + 256]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    net = UNet(CH, 1).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        p = net(torch.as_tensor(x)[None, None].to(dev)).cpu().numpy()[0, 0]
    preds = dict(noisy=x, bm3d=bm3d_anscombe(x, TUNED["bm3d"]),
                 nlm=nlm(x, TUNED["nlm"]), tv=tv(x, TUNED["tv"]), unet=p)
    out = {k: [float(psnr(gt, np.clip(im, 0, 1), data_range=1)),
               float(ssim(gt, np.clip(im, 0, 1), data_range=1))]
           for k, im in preds.items()}
    return int(stem), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/real_large_fair.json")
    ap.add_argument("--ckpt", default="checkpoints/real_unet.pt")
    ap.add_argument("--ch", type=int, default=32)
    a = ap.parse_args()
    global CKPT, CH
    CKPT, CH = a.ckpt, a.ch

    from real_common import pairs
    tasks = []
    for npath, gpath, stem in pairs("large"):
        n_shape = (1024, 1024)
        ps = [(i, j) for i in range(0, n_shape[0] - 256 + 1, 256)
              for j in range(0, n_shape[1] - 256 + 1, 256)]
        step = max(1, len(ps) // a.n_patch)
        for i, j in ps[::step][:a.n_patch]:
            tasks.append((stem, i, j, npath, gpath))
    print("patch tasks", len(tasks), "workers", a.workers, flush=True)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(work, tasks)

    by = {}
    for stem, out in res:
        by.setdefault(stem, []).append(out)
    rows = []
    for stem in sorted(by):
        d = by[stem]
        rows.append({k: [float(np.mean([o[k][0] for o in d])),
                         float(np.mean([o[k][1] for o in d]))] for k in d[0]})
        print(stem, {k: round(v[0], 2) for k, v in rows[-1].items()}, flush=True)
    mean = {k: [float(np.mean([r[k][0] for r in rows])),
                float(np.mean([r[k][1] for r in rows]))] for k in rows[0]}
    print("means", {k: round(v[0], 2) for k, v in mean.items()}, flush=True)
    print("ssim", {k: round(v[1], 3) for k, v in mean.items()}, flush=True)
    json.dump(dict(tuned=TUNED, rows=rows), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
