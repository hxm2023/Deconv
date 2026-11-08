"""Near-optimality gate for task/dose conditioning.

If the acquisition task and the dose are already identifiable from the noisy
image, explicit task labels or language/soft-token conditioning have little
headroom: a blind restoration model can infer the same information internally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dataset_paths, degrade, load_image, split_paths

TASKS = ["bbbc014", "bbbc015", "bbbc016", "bbbc018"]
DOSES = [3, 10, 30, 100, 300]


def fov_paths(task, n=20, seed=1):
    paths, _ = dataset_paths(task)
    if task == "bbbc014":
        return split_paths(paths, seed=0, n_val=20, n_test=20)["test"][:n]
    return split_paths(paths, seed=seed, n_val=0, n_test=n)["test"][:n]


def make_patches(paths, dose, n_patch=20, size=128, seed=0):
    rng = np.random.default_rng(seed)
    X, y_fov = [], []
    for k, p in enumerate(paths):
        x = load_image(p, 255.0)
        if x.shape[0] < size or x.shape[1] < size:
            continue
        for _ in range(n_patch):
            i = int(rng.integers(0, x.shape[0] - size + 1))
            j = int(rng.integers(0, x.shape[1] - size + 1))
            patch = x[i:i + size, j:j + size]
            X.append(degrade(patch, photons=dose, rng=rng))
            y_fov.append(k)
    return np.asarray(X, dtype=np.float32), np.asarray(y_fov)


def features(X):
    from scipy.stats import kurtosis, skew
    feats = []
    for x in X:
        v = x.ravel()
        gx = np.diff(x, axis=1).ravel()
        gy = np.diff(x, axis=0).ravel()
        lap = (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]
               - 4 * x[1:-1, 1:-1]).ravel()
        f = [v.mean(), v.std(), skew(v), kurtosis(v),
             np.percentile(v, 1), np.percentile(v, 5), np.percentile(v, 25),
             np.percentile(v, 50), np.percentile(v, 75),
             np.percentile(v, 95), np.percentile(v, 99),
             np.abs(gx).mean(), np.abs(gy).mean(), lap.var()]
        spec = np.abs(np.fft.rfft2(x - x.mean())) ** 2
        h, w = spec.shape
        for lo, hi in [(1, 8), (8, 24), (24, 64)]:
            f.append(float(spec[lo:hi, :].mean()))
        feats.append(f)
    return np.asarray(feats, dtype=np.float32)


def linear_probe(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(sc.transform(Xtr), ytr)
    acc = float((clf.predict(sc.transform(Xte)) == yte).mean())
    return acc


def cnn_probe(Xtr, ytr, Xte, yte, n_class, epochs=12):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.fc = nn.Linear(64, n_class)

        def forward(self, x):
            x = self.c(x)
            return self.fc(x.mean(dim=(2, 3)))

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt = torch.as_tensor(Xtr)[:, None].to(dev)
    yt = torch.as_tensor(ytr).to(dev)
    xe = torch.as_tensor(Xte)[:, None].to(dev)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for q in range(0, len(xt), 64):
            idx = perm[q:q + 64]
            loss = F.cross_entropy(net(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(xe).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=int, default=30)
    ap.add_argument("--n-patch", type=int, default=20)
    ap.add_argument("--out", default="results/probe_task_dose.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    # task probe at fixed dose: 10 FOVs per task train, 10 test
    Xtr, ytr, Xte, yte = [], [], [], []
    for k, task in enumerate(TASKS):
        ps = fov_paths(task)
        for split, X, y in [(ps[:10], Xtr, ytr), (ps[10:20], Xte, yte)]:
            x, _ = make_patches(split, a.dose, n_patch=a.n_patch,
                                seed=int(rng.integers(1e6)))
            X.append(x); y.extend([k] * len(x))
    Xtr = np.concatenate(Xtr); Xte = np.concatenate(Xte)
    ytr = np.asarray(ytr); yte = np.asarray(yte)
    task_lin = linear_probe(features(Xtr), ytr, features(Xte), yte)
    task_cnn = cnn_probe(Xtr, ytr, Xte, yte, len(TASKS))
    task_n = (len(ytr), len(yte))
    print("task probe: linear", round(task_lin, 3), "cnn", round(task_cnn, 3),
          "n", task_n, flush=True)

    # dose probe on BBBC014 test FOVs: 10 FOVs train, 10 test
    ps = fov_paths("bbbc014")
    Xtr, ytr, Xte, yte = [], [], [], []
    for d_i, dose in enumerate(DOSES):
        for split, X, y in [(ps[:10], Xtr, ytr), (ps[10:20], Xte, yte)]:
            x, _ = make_patches(split, dose, n_patch=a.n_patch,
                                seed=int(rng.integers(1e6)))
            X.append(x); y.extend([d_i] * len(x))
    Xtr = np.concatenate(Xtr); Xte = np.concatenate(Xte)
    ytr = np.asarray(ytr); yte = np.asarray(yte)
    dose_lin = linear_probe(features(Xtr), ytr, features(Xte), yte)
    dose_cnn = cnn_probe(Xtr, ytr, Xte, yte, len(DOSES))
    dose_n = (len(ytr), len(yte))
    print("dose probe: linear", round(dose_lin, 3), "cnn", round(dose_cnn, 3),
          "n", dose_n, flush=True)

    out = dict(task=dict(linear=task_lin, cnn=task_cnn,
                         n_train=task_n[0], n_test=task_n[1]),
               dose=dict(linear=dose_lin, cnn=dose_cnn,
                         n_train=dose_n[0], n_test=dose_n[1]),
               dose_used=a.dose, tasks=TASKS, doses=DOSES)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
