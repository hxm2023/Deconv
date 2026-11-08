"""Re-evaluate only the U-Net on real pairs, reusing validated classical rows."""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CKPT = "checkpoints/real_unet.pt"
CH = 32


def work(item):
    import torch
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

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
    p = np.clip(u, 0, 1)
    return str(stem), [float(psnr(g, p, data_range=1)),
                       float(ssim(g, p, data_range=1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=243)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt", default="checkpoints/real_unet.pt")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--base", default="results/real_affine_test_all.json")
    ap.add_argument("--out", default="results/real_affine_test_all_strong.json")
    a = ap.parse_args()
    global CKPT, CH
    CKPT, CH = a.ckpt, a.ch
    from real_common import split_small
    _, val, test = split_small(0)
    test = test[:a.n_test]
    base = json.load(open(a.base))
    print("tasks", len(test), flush=True)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = dict(pool.map(work, test))
    missing = [r["stem"] for r in base["rows"] if r["stem"] not in res]
    if missing:
        raise RuntimeError(f"missing stems: {missing[:5]}")

    rows = []
    for r in base["rows"]:
        rr = dict(r)
        rr["unet"] = res[r["stem"]]
        rows.append(rr)
    summary = dict(base["summary"])
    summary["unet"] = [float(np.mean([r["unet"][0] for r in rows])),
                       float(np.mean([r["unet"][1] for r in rows]))]
    print("unet", round(summary["unet"][0], 2),
          round(summary["unet"][1], 3), flush=True)
    d = np.array([r["unet"][0] - r["bm3d"][0] for r in rows])
    from scipy.stats import wilcoxon
    print("gain", round(float(d.mean()), 2), "wins", int((d > 0).sum()),
          "p", float(wilcoxon(d).pvalue), flush=True)
    out = dict(tuned=base["tuned"], affine=base["affine"], rows=rows,
               summary=summary, ckpt=a.ckpt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
