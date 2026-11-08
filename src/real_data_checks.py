"""Archive the real-data quantity checks cited in the SI.

- ground-truth standard deviation of the 40 held-out pairs;
- thumbnail correlation between held-out and training images (leakage check);
- correlation of a small image with tiles of the same-ID large field.
"""
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
    return (x - x.mean()) / (x.std() + 1e-6)


def main():
    train, val, test = split_small(0)
    rng = np.random.default_rng(1)
    train_sample = list(rng.choice(len(train), size=min(400, len(train)),
                                   replace=False))
    tr = [thumb(train[i][1]) for i in train_sample]

    def max_corr(items):
        out = []
        for _, gp, _ in items:
            v = thumb(gp)
            out.append(max(float((v * t).mean()) for t in tr))
        return out

    test_corr = max_corr(test)
    val_corr = max_corr(val[:20])
    test_std = [(int(stem), float(norm99(io.imread(gp).astype(np.float32)).std()))
                for _, gp, stem in test]
    test_std.sort(key=lambda x: x[1])
    res = json.load(open("results/real_small_test_all.json"))["rows"]
    std_by_stem = {s: v for s, v in test_std}
    loss = [(int(t[2]), std_by_stem[int(t[2])], float(r["unet"][0]),
             float(r["bm3d"][0]))
            for t, r in zip(test, res) if r["unet"][0] <= r["bm3d"][0]]

    small_large = []
    root = Path("datasets/real_nuclei")
    for stem in range(1000, 1020):
        sp = root / "small_images" / "gt" / f"{stem}.tif"
        lp = root / "large_images" / "gt" / f"{stem}.tif"
        if not sp.exists() or not lp.exists():
            continue
        s = norm99(io.imread(sp).astype(np.float32))
        L = norm99(io.imread(lp).astype(np.float32))
        s = (s - s.mean()) / (s.std() + 1e-6)
        best = -2.0
        for i in range(0, L.shape[0] - 256 + 1, 64):
            for j in range(0, L.shape[1] - 256 + 1, 64):
                t = L[i:i + 256, j:j + 256]
                t = (t - t.mean()) / (t.std() + 1e-6)
                best = max(best, float((s * t).mean()))
        small_large.append((int(stem), best))

    out = dict(
        n_test=len(test),
        test_gt_std_min=test_std[0][1],
        test_gt_std_min_stem=test_std[0][0],
        test_gt_std_max=test_std[-1][1],
        losses=loss,
        test_thumbnail_max=float(np.max(test_corr)),
        val_thumbnail_max=float(np.max(val_corr)),
        n_train_thumbs=len(tr),
        small_large_max_corr=float(max(v for _, v in small_large)),
        small_large=small_large,
    )
    Path("results").mkdir(exist_ok=True)
    Path("results/real_data_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
