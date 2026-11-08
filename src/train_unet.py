"""Train a U-Net for low-dose deconvolution with strict FOV-level splits."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (airy_psf, bbbc039_official_split, dataset_paths, degrade,
                    load_image, patches, psnr, save_json, split_paths)
from model import UNet

def scramble_image(x, rng):
    """Random-phase surrogate with the same power spectrum as x."""
    f = np.fft.fft2(x)
    noise = rng.normal(size=x.shape)
    fn = np.fft.fft2(noise)
    phase = np.exp(1j * np.angle(fn))
    out = np.real(np.fft.ifft2(np.abs(f) * phase))
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out.astype(np.float32)


DOSES_ALL = [3.0, 10.0, 30.0, 100.0, 300.0]
SIGMAS_ALL = [1.5, 2.0, 2.5]


def cond_value(photons):
    lo, hi = math.log10(DOSES_ALL[0]), math.log10(DOSES_ALL[-1])
    return (math.log10(photons) - lo) / (hi - lo)


def sigma_value(sigma):
    return (sigma - 1.5) / 1.0


def build_training_set(fovs, n_patches, patch, sigma, photons, dose_cond,
                       sigma_cond, rng, mixed_doses=False, psf_mix=False):
    X, Y, C = [], [], []
    n = len(fovs)
    airy = airy_psf() if psf_mix else None
    for k in range(n_patches):
        img = fovs[k % n]
        h, w = img.shape
        i = int(rng.integers(0, h - patch + 1))
        j = int(rng.integers(0, w - patch + 1))
        gt = img[i:i + patch, j:j + patch]
        p = (float(rng.choice(DOSES_ALL)) if (dose_cond or mixed_doses)
             else photons)
        s = float(rng.choice(SIGMAS_ALL)) if sigma_cond else sigma
        psf = airy if (psf_mix and rng.random() < 0.5) else None
        y = degrade(gt, s, p, rng, psf=psf)
        X.append(y)
        Y.append(gt)
        c = []
        if dose_cond:
            c.append(cond_value(p))
        if sigma_cond:
            c.append(sigma_value(s))
        C.append(c)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)[:, None]
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)[:, None]
    if C:
        c = np.array(C, dtype=np.float32, copy=True)
        c = np.broadcast_to(c[:, :, None, None], (len(c), c.shape[1], patch, patch))
        C = torch.as_tensor(np.ascontiguousarray(c), dtype=torch.float32)
    else:
        C = None
    return X, Y, C


def ingest(x, c, dev):
    if c is None:
        return x.to(dev)
    return torch.cat([x.to(dev), c.to(dev)], 1)


def eval_val(net, fovs, sigma, photons, dose_cond, sigma_cond, patch, dev,
             batch=64, max_per_fov=4, val_all_doses=False, psf_mix=False):
    net.eval()
    per = {}
    doses = DOSES_ALL if (dose_cond or val_all_doses) else [photons]
    sigmas = SIGMAS_ALL if sigma_cond else [sigma]
    psfs = [("gauss", None), ("airy", airy_psf())] if psf_mix else [("gauss", None)]
    with torch.no_grad():
      for pht in doses:
        for sig in sigmas:
          for psf_name, psf in psfs:
                rng = np.random.default_rng(999)
                gts, ys = [], []
                for img in fovs:
                    ps = list(patches(img, patch))
                    step = max(1, len(ps) // max_per_fov)
                    for _, _, gt in ps[::step][:max_per_fov]:
                        gts.append(gt)
                        ys.append(degrade(gt, sig, pht, rng, psf=psf))
                preds = []
                for q in range(0, len(ys), batch):
                    xb = torch.as_tensor(np.asarray(ys[q:q + batch]),
                                         dtype=torch.float32)[:, None]
                    cs = []
                    if dose_cond:
                        cs.append(cond_value(pht))
                    if sigma_cond:
                        cs.append(sigma_value(sig))
                    if cs:
                        h, w = xb.shape[-2:]
                        c = torch.as_tensor(np.asarray(cs, dtype=np.float32))[None, :, None, None]
                        c = c.expand(len(xb), -1, h, w).contiguous()
                        xb = torch.cat([xb, c], 1)
                    preds.append(net(xb.to(dev)).cpu().numpy()[:, 0])
                preds = np.concatenate(preds, 0)
                vals = [psnr(g, p) for g, p in zip(gts, preds)]
                key = f"p{pht:g}_s{sig:g}" + (f"_{psf_name}" if psf_mix else "")
                per[key] = float(np.mean(vals))
    net.train()
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=8000)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--photons", type=float, default=300.0)
    ap.add_argument("--doses", default="3,10,30,100,300",
                    help="comma-separated mixed-dose set")
    ap.add_argument("--dose-cond", action="store_true")
    ap.add_argument("--sigma-cond", action="store_true")
    ap.add_argument("--scramble", action="store_true",
                    help="phase-scramble training FOVs (negative control)")
    ap.add_argument("--mixed-doses", action="store_true",
                    help="sample all doses even without a dose channel (blind control)")
    ap.add_argument("--psf-mix", action="store_true",
                    help="train on a 50/50 Gaussian/Airy PSF mixture")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="bbbc014")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--val-patches", type=int, default=4)
    ap.add_argument("--val-all-doses", action="store_true")
    ap.add_argument("--out", default="checkpoints/unet_d300.pt")
    a = ap.parse_args()
    global DOSES_ALL
    DOSES_ALL = [float(x) for x in a.doses.split(",")]

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)

    paths, scale = dataset_paths(a.dataset, a.channel)
    if a.dataset == "bbbc039":
        sp = bbbc039_official_split()
    else:
        sp = split_paths(paths, a.split_seed, a.n_val, a.n_test)
    fovs_train = [load_image(p, scale) for p in sp["train"]]
    fovs_val = [load_image(p, scale) for p in sp["val"]]
    if a.scramble:
        srng = np.random.default_rng(12345)
        fovs_train = [scramble_image(x, srng) for x in fovs_train]
        print("training FOVs are phase-scrambled (negative control)", flush=True)
    print(f"dataset {a.dataset} channel {a.channel} scale {scale} "
          f"train fovs {len(fovs_train)} val fovs {len(fovs_val)} "
          f"test fovs {len(sp['test'])}", flush=True)

    rng = np.random.default_rng(a.seed)
    X, Y, C = build_training_set(
        fovs_train, a.patches, a.patch, a.sigma, a.photons,
        a.dose_cond, a.sigma_cond, rng, mixed_doses=a.mixed_doses,
        psf_mix=a.psf_mix)
    print("patches", tuple(X.shape), "cond", None if C is None else tuple(C.shape), flush=True)

    in_ch = 1 + (1 if a.dose_cond else 0) + (1 if a.sigma_cond else 0)
    net = UNet(a.ch, in_ch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best = -1.0
    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        idx = torch.randperm(len(X))
        tot = 0.0
        for q in range(0, len(X), a.batch):
            mb = idx[q:q + a.batch]
            xb = X[mb]
            # flip/rot90 augmentation on the pair
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                xb = torch.rot90(xb, k, dims=(2, 3))
                yb = torch.rot90(Y[mb], k, dims=(2, 3))
            else:
                yb = Y[mb]
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=(3,))
                yb = torch.flip(yb, dims=(3,))
            xb = ingest(xb, None if C is None else C[mb], dev)
            yb = yb.to(dev)
            pred = net(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(mb)
        sched.step()
        v, per = eval_val(net, fovs_val, a.sigma, a.photons, a.dose_cond,
                          a.sigma_cond, 128, dev, max_per_fov=a.val_patches,
                          val_all_doses=a.val_all_doses, psf_mix=a.psf_mix)
        hist.append(dict(epoch=ep, loss=tot / len(X), val_psnr=v, per=per))
        print(f"epoch {ep} loss {tot/len(X):.5f} val_psnr {v:.3f} "
              f"per { {k: round(x,2) for k,x in per.items()} } "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v > best:
            best = v
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=net.state_dict(), in_ch=in_ch,
                            dose_cond=a.dose_cond, sigma_cond=a.sigma_cond,
                            sigma=a.sigma, photons=a.photons,
                            val_psnr=v, args=vars(a)), a.out)
            print("  saved", a.out, "best", round(best, 3), flush=True)
    save_json(Path(a.out).with_suffix(".json"),
              dict(best_val_psnr=best, history=hist, args=vars(a),
                   split={k: [Path(x).stem for x in v] for k, v in sp.items()}))


if __name__ == "__main__":
    main()
