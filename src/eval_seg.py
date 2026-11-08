"""Evaluate task-aware models and classical baselines on BBBC039 segmentation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv, tv_deconv
from common import bbbc039_official_split, degrade, save_json
from seg_common import dice, iou, load_pair, psnr, ssim
from seg_model import MultiTaskUNet

DOSES = [3, 30, 300]


def otsu_mask(img):
    try:
        thr = threshold_otsu(img)
    except Exception:
        thr = float(img.mean())
    return img > thr


def model_pred(net, y, dev):
    with torch.no_grad():
        xt = torch.as_tensor(y, dtype=torch.float32)[None, None].to(dev)
        rest, seg = net(xt)
    r = None if rest is None else rest.cpu().numpy()[0, 0]
    s = None if seg is None else (torch.sigmoid(seg).cpu().numpy()[0, 0] > 0.5)
    return r, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-patch", type=int, default=10)
    ap.add_argument("--out", default="results/seg_eval.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = ck["mode"]
    net = MultiTaskUNet(32, 1, restore=mode in ("joint", "rest"),
                        segment=mode in ("joint", "seg"),
                        guided=ck.get("guided", False)).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    sp = bbbc039_official_split()
    test = sp["test"][::max(1, len(sp["test"]) // a.n_test)][:a.n_test]
    out = {}
    for P in DOSES:
        rows = {}
        for path in test:
            img, mask = load_pair(path)
            ps = list(__import__("common").patches(img, 128))
            step = max(1, len(ps) // a.n_patch)
            ps = ps[::step][:a.n_patch]
            vals = {k: [] for k in ["noisy", "tvden", "tvdec", "rest_otsu",
                                    "seg", "joint_seg", "joint_rest_otsu"]}
            for i, j, gt, mt in [(i, j, img[i:i+128, j:j+128],
                                  mask[i:i+128, j:j+128]) for i, j, _ in ps]:
                rng = np.random.default_rng(2026)
                y = degrade(gt, 2.0, P, rng)
                preds = dict(
                    noisy=y, tvden=tv(y, 0.04),
                    tvdec=tv_deconv(y, 2, 0.04, iters=30))
                rest, seg = model_pred(net, y, dev)
                if rest is not None:
                    preds["rest"] = rest
                    preds["joint_rest"] = rest
                for k, im in [("noisy", y), ("tvden", preds["tvden"]),
                              ("tvdec", preds["tvdec"])]:
                    vals[k].append(dice(mt, otsu_mask(im)))
                if rest is not None:
                    vals["rest_otsu"].append(dice(mt, otsu_mask(rest)))
                    vals["joint_rest_otsu"].append(dice(mt, otsu_mask(rest)))
                if seg is not None:
                    vals["seg"].append(dice(mt, seg))
                    vals["joint_seg"].append(dice(mt, seg))
            for k in vals:
                if vals[k]:
                    rows.setdefault(k, []).append(float(np.mean(vals[k])))
        out[str(P)] = {k: dict(dice=float(np.mean(v)), n=len(v))
                       for k, v in rows.items() if v}
        print(P, out[str(P)], flush=True)
    save_json(a.out, out)


if __name__ == "__main__":
    main()
