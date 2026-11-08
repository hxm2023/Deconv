"""Train a physics-informed unrolled deconvolution network."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (dataset_paths, load_image, save_json, split_paths)
from model_unrolled import UnrolledDeconv
from train_unet import (DOSES_ALL, build_training_set, cond_value, eval_val,
                        ingest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=12000)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-patches", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/bbbc014_unrolled.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)
    paths, scale = dataset_paths("bbbc014")
    sp = split_paths(paths, 0, 20, 20)
    fovs_train = [load_image(p, scale) for p in sp["train"]]
    fovs_val = [load_image(p, scale) for p in sp["val"]]
    rng = np.random.default_rng(a.seed)
    X, Y, C = build_training_set(fovs_train, a.patches, a.patch, 2.0, 300.0,
                                 True, False, rng)
    print("patches", tuple(X.shape), flush=True)
    net = UnrolledDeconv(a.ch, a.K).to(dev)
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
            xb = ingest(X[mb], C[mb], dev)
            yb = Y[mb].to(dev)
            pred = net(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(mb)
        sched.step()
        v, per = eval_val(net, fovs_val, 2.0, 300.0, True, False, 128, dev,
                          max_per_fov=a.val_patches)
        hist.append(dict(epoch=ep, loss=tot / len(X), val_psnr=v,
                         per=per))
        print(f"epoch {ep} loss {tot/len(X):.5f} val {v:.3f} "
              f"per { {k: round(x,2) for k,x in per.items()} } "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v > best:
            best = v
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=net.state_dict(), model_type="unrolled",
                            in_ch=2, K=a.K, ch=a.ch, val_psnr=v,
                            args=vars(a)), a.out)
            print("  saved", a.out, flush=True)
    save_json(Path(a.out).with_suffix(".json"),
              dict(best=best, history=hist, args=vars(a)))


if __name__ == "__main__":
    main()
