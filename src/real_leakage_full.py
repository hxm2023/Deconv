"""Full thumbnail leakage audit: all 243 test vs all 1,951 train images."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from skimage import io, transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_common import norm99, split_small


def thumb(path, n=64):
    x = io.imread(path).astype(np.float32)
    x = transform.resize(x, (n, n), anti_aliasing=True)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x.ravel()


def main():
    train, _, test = split_small(0)
    tr = np.asarray([thumb(p[1]) for p in train], dtype=np.float32)
    te = np.asarray([thumb(p[1]) for p in test], dtype=np.float32)
    C = te @ tr.T / tr.shape[1]
    best = C.argmax(axis=1)
    mx = C[np.arange(len(te)), best]
    top = int(mx.argmax())
    tp, tn = test[top], train[best[top]]
    a = norm99(io.imread(tp[1]).astype(np.float32))
    b = norm99(io.imread(tn[1]).astype(np.float32))
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    full_corr = float((a * b).mean())
    out = dict(
        n_train=len(train), n_test=len(test),
        max_thumb_corr=float(mx.max()),
        mean_max_thumb_corr=float(mx.mean()),
        n_above_0_8=int((mx > 0.8).sum()),
        n_above_0_9=int((mx > 0.9).sum()),
        top_test_stem=int(tp[2]), top_train_stem=int(tn[2]),
        top_full_corr=full_corr,
    )
    print(out, flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/real_leakage_full.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
