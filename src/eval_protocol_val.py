"""Validation-selected protocol doses for the strong model and fair TV."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import tv, tv_deconv
from common import (dataset_paths, degrade, fg_psnr, load_image, patches,
                    psnr, ssim, split_paths)
from model import UNet

torch.set_num_threads(2)
DOSES = [3, 10, 30, 100, 300]
PARAMS = {3: (0.20, 0.3, 30), 10: (0.08, 0.3, 30), 30: (0.04, 0.3, 30),
          100: (0.02, 3.0, 30), 300: (0.01, 3.0, 30)}
DN_W = {3: 0.4, 10: 0.3, 30: 0.12, 100: 0.04, 300: 0.02}
TARGETS = [("psnr", t) for t in (30, 32, 34)] + \
          [("ssim", t) for t in (0.85, 0.90, 0.92)] + \
          [("fg", t) for t in (22, 24, 26)]


def sample(img, n=10):
    ps = list(patches(img, 128))
    step = max(1, len(ps) // n)
    return [gt for _, _, gt in ps[::step][:n]]


def proto(curves, target):
    for d in DOSES:
        if float((curves[d] >= target).mean()) >= 0.9:
            return d
    return None


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("checkpoints/bbbc014_blind_strong.pt", map_location=dev,
                    weights_only=False)
    ch = ck.get("args", {}).get("ch", 64)
    net = UNet(ch, ck["in_ch"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    val = [load_image(p, scale) for p in sp["val"]]
    gts = [sample(img) for img in val]
    out = {}
    for dose in DOSES:
        lam, rho, it = PARAMS[dose]
        rows = []
        for fi in range(len(val)):
            rng = np.random.default_rng(2026)
            ys = [degrade(g, 2.0, dose, rng) for g in gts[fi]]
            x = torch.as_tensor(np.asarray(ys))[:, None].to(dev)
            with torch.no_grad():
                p = net(x).cpu().numpy()[:, 0]
            p_tv = [tv_deconv(y, 2.0, lam, rho=rho, iters=it, pad=32)
                    for y in ys]
            p_dn = [tv(y, DN_W[dose]) for y in ys]
            rows.append(dict(
                unet_psnr=[psnr(g, q) for g, q in zip(gts[fi], p)],
                unet_ssim=[ssim(g, q) for g, q in zip(gts[fi], p)],
                unet_fg=[fg_psnr(g, q) for g, q in zip(gts[fi], p)],
                tv_deconv=[psnr(g, q) for g, q in zip(gts[fi], p_tv)],
                tv_denoise=[psnr(g, q) for g, q in zip(gts[fi], p_dn)],
                tv_deconv_ssim=[ssim(g, q) for g, q in zip(gts[fi], p_tv)],
                tv_denoise_ssim=[ssim(g, q) for g, q in zip(gts[fi], p_dn)],
                tv_deconv_fg=[fg_psnr(g, q) for g, q in zip(gts[fi], p_tv)],
                tv_denoise_fg=[fg_psnr(g, q) for g, q in zip(gts[fi], p_dn)],
            ))
        out[str(dose)] = dict(lam=lam, rho=rho, iters=it, rows=rows)
        print(dose, "U", round(float(np.mean([np.mean(r["unet_psnr"])
                                             for r in rows])), 2),
              "TV", round(float(np.mean([np.mean(r["tv_deconv"])
                                        for r in rows])), 2), flush=True)

    B = json.load(open("results/blind_fair_doses.json"))
    D = json.load(open("results/tvdeconv_doses.json"))
    test_u = {m: {d: np.array([r[m] for r in B[str(d)]]) for d in DOSES}
              for m in ("psnr", "ssim", "fg")}
    test_tv = {}
    for m, kd, kn in [("psnr", "tv_deconv", "tv_denoise"),
                      ("ssim", "tv_deconv_ssim", "tv_denoise_ssim"),
                      ("fg", "tv_deconv_fg", "tv_denoise_fg")]:
        test_tv[m] = {}
        for d in DOSES:
            rows = D[str(d)]["rows"]
            a = np.array([np.mean(r[kd]) for r in rows])
            b = np.array([np.mean(r[kn]) for r in rows])
            test_tv[m][d] = a if a.mean() >= b.mean() else b

    def val_curve(m, which):
        if which == "unet":
            key = "unet_" + m
        else:
            key = which if m == "psnr" else which + "_" + m
        return {d: np.array([np.mean(r[key]) for r in out[str(d)]["rows"]])
                for d in DOSES}

    res = []
    for m, t in TARGETS:
        ud = proto(val_curve(m, "unet"), t)
        td = proto(val_curve(m, "tv_deconv"), t)
        u_test = float((test_u[m][ud] >= t).mean()) if ud else None
        t_test = float((test_tv[m][td] >= t).mean()) if td else None
        res.append(dict(metric=m, target=t, unet_dose=ud, tv_dose=td,
                        factor=(None if not (ud and td) else td / ud),
                        unet_test_success=u_test, tv_test_success=t_test))
        print(m, t, "ud", ud, "td", td,
              "Utest", None if u_test is None else round(u_test, 2),
              "TVtest", None if t_test is None else round(t_test, 2),
              flush=True)
    Path("results/protocol_val.json").write_text(json.dumps(
        dict(targets=res, rows=out), indent=2))


if __name__ == "__main__":
    main()
