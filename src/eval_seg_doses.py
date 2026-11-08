"""Downstream nuclei segmentation Dice versus photon dose."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv, tv_deconv
from common import bbbc039_official_split, degrade, save_json
from seg_common import dice, load_pair
from seg_model import MultiTaskUNet

DOSES = [1, 2, 3, 5, 10, 20, 30, 50, 100, 300]


def otsu(img):
    try:
        return img > threshold_otsu(img)
    except Exception:
        return img > float(img.mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/seg_rest.pt")
    ap.add_argument("--doses", default="1,2,3,5,10,20,30,50,100,300")
    ap.add_argument("--out", default="results/seg_doses.json")
    a = ap.parse_args()
    doses = [float(x) for x in a.doses.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev,
                    weights_only=False)
    net = MultiTaskUNet(32, 1, restore=True, segment=False,
                        guided=False).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    sp = bbbc039_official_split()
    test = sp["test"][::max(1, len(sp["test"]) // 20)][:20]
    out = {}
    for P in doses:
        rows = {k: [] for k in ["noisy", "tvden", "tvdec", "unet"]}
        for path in test:
            img, mask = load_pair(path)
            ps = list(__import__("common").patches(img, 128))
            step = max(1, len(ps) // 10)
            ps = ps[::step][:10]
            vals = {k: [] for k in rows}
            for i, j, _ in ps:
                gt = img[i:i + 128, j:j + 128]
                mt = mask[i:i + 128, j:j + 128]
                rng = np.random.default_rng(2026)
                y = degrade(gt, 2.0, P, rng)
                with torch.no_grad():
                    xt = torch.as_tensor(y, dtype=torch.float32)[None, None].to(dev)
                    rest, _ = net(xt)
                preds = dict(noisy=y, tvden=tv(y, 0.04),
                             tvdec=tv_deconv(y, 2, 0.04, iters=30),
                             unet=rest.cpu().numpy()[0, 0])
                for k, im in preds.items():
                    vals[k].append(dice(mt, otsu(im)))
            for k in rows:
                rows[k].append(float(np.mean(vals[k])))
        out[str(P)] = {k: float(np.mean(v)) for k, v in rows.items()}
        print(P, {k: round(v, 3) for k, v in out[str(P)].items()}, flush=True)
    save_json(a.out, out)


if __name__ == "__main__":
    main()
