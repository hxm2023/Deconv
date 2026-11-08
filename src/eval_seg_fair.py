"""Fair BBBC039 segmentation comparison with per-dose tuned TV and all test FOVs."""
from __future__ import annotations

import json
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

DOSES = [1, 3, 10, 30, 100, 300]


def otsu(img):
    try:
        return img > threshold_otsu(img)
    except Exception:
        return img > float(img.mean())


def matched(curve, target):
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
    tune = json.load(open("results/seg_tv_tuning.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/seg_rest_p1.pt", map_location=dev,
                    weights_only=False)
    net = MultiTaskUNet(32, 1, restore=True, segment=False).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    sp = bbbc039_official_split()
    test = sp["test"]
    out = {}
    for P in DOSES:
        lam = float(tune[str(P)]["tvdec_w"])
        w = float(tune[str(P)]["tvden_w"])
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
                preds = dict(
                    noisy=y, tvden=tv(y, w),
                    tvdec=tv_deconv(y, 2, lam, iters=30),
                    unet=rest.cpu().numpy()[0, 0])
                for k, im in preds.items():
                    vals[k].append(dice(mt, otsu(im)))
            for k in rows:
                rows[k].append(float(np.mean(vals[k])))
        out[str(P)] = {k: float(np.mean(v)) for k, v in rows.items()}
        out[str(P)]["_rows"] = rows
        print(P, {k: round(v, 3) for k, v in out[str(P)].items()
                  if not k.startswith("_")}, flush=True)
    save_json("results/seg_fair.json", dict(tuning=tune, data=out))


if __name__ == "__main__":
    main()
