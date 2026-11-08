"""Train task-aware low-dose deconvolution + nuclei segmentation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bbbc039_official_split, degrade, save_json
from seg_common import dice, load_pair, psnr, random_pair_patch, eval_patches
from seg_model import MultiTaskUNet

DOSES = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 100.0, 300.0]


def seg_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    d = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + d


def build_set(paths, n, size, rng):
    pairs = [load_pair(p) for p in paths]
    X, Y, M = [], [], []
    for k in range(n):
        img, mask = pairs[k % len(pairs)]
        gt, mt = random_pair_patch(img, mask, size, rng)
        p = float(rng.choice(DOSES))
        X.append(degrade(gt, 2.0, p, rng))
        Y.append(gt)
        M.append(mt)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)[:, None]
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)[:, None]
    M = torch.as_tensor(np.asarray(M), dtype=torch.float32)[:, None]
    return X, Y, M


def evaluate(net, paths, mode, dev, size=128, n_patch=4):
    net.eval()
    per = {}
    with torch.no_grad():
        for p in DOSES:
            rng = np.random.default_rng(999)
            rest_vals, seg_vals = [], []
            for path in paths:
                img, mask = load_pair(path)
                pp = eval_patches(img, mask, size, n_patch)
                for _, _, gt, mt in pp:
                    y = degrade(gt, 2.0, p, rng)
                    xt = torch.as_tensor(y, dtype=torch.float32)[None, None].to(dev)
                    rest, seg = net(xt)
                    if rest is not None:
                        r = rest.cpu().numpy()[0, 0]
                        rest_vals.append(psnr(gt, r))
                    if seg is not None:
                        s = (torch.sigmoid(seg).cpu().numpy()[0, 0] > 0.5)
                        seg_vals.append(dice(mt, s))
            per[str(int(p))] = dict(
                psnr=float(np.mean(rest_vals)) if rest_vals else float("nan"),
                dice=float(np.mean(seg_vals)) if seg_vals else float("nan"))
    net.train()
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seg", "joint", "rest"], default="joint")
    ap.add_argument("--patches", type=int, default=12000)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guided", action="store_true")
    ap.add_argument("--out", default="checkpoints/seg_joint.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)
    sp = bbbc039_official_split()
    rng = np.random.default_rng(a.seed)
    X, Y, M = build_set(sp["train"], a.patches, a.patch, rng)
    print("patches", tuple(X.shape), flush=True)
    net = MultiTaskUNet(32, 1, restore=a.mode in ("joint", "rest"),
                        segment=a.mode in ("joint", "seg"),
                        guided=a.guided).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best = -1e9
    hist = []
    t0 = time.time()
    for ep in range(a.epochs):
        idx = torch.randperm(len(X))
        tot = 0.0
        for q in range(0, len(X), a.batch):
            mb = idx[q:q + a.batch]
            xb = X[mb].to(dev); yb = Y[mb].to(dev); mbt = M[mb].to(dev)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                xb = torch.rot90(xb, k, dims=(2, 3))
                yb = torch.rot90(yb, k, dims=(2, 3))
                mbt = torch.rot90(mbt, k, dims=(2, 3))
            rest, seg = net(xb)
            loss = 0.0
            if rest is not None:
                loss = loss + F.mse_loss(rest, yb)
            if seg is not None:
                loss = loss + a.lam * seg_loss(seg, mbt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(mb)
        sched.step()
        v = evaluate(net, sp["val"], a.mode, dev)
        score = (v["3"]["dice"] if a.mode in ("joint", "seg") else 0.0) + \
                0.001 * (v["3"]["psnr"] if a.mode in ("joint", "rest") else 0.0)
        hist.append(dict(epoch=ep, loss=tot / len(X), val=v, score=score))
        print(f"epoch {ep} loss {tot/len(X):.4f} "
              f"P3 {v['3']} P30 {v['30']} P300 {v['300']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if score > best:
            best = score
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=net.state_dict(), mode=a.mode,
                            guided=a.guided, in_ch=1, val=v, args=vars(a)), a.out)
            print("  saved", a.out, flush=True)
    save_json(Path(a.out).with_suffix(".json"),
              dict(best=best, history=hist, args=vars(a)))


if __name__ == "__main__":
    main()
