"""Shared data, degradation, and metrics for the low-dose microscopy benchmark.

Splits are by field of view (FOV). A FOV used for training is never used for
validation, test, or baseline tuning.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve
from skimage import io
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RESULTS = ROOT / "results"


def bbbc039_paths():
    fs = sorted(glob.glob(str(DATA / "images" / "*.tif")))
    if not fs:
        raise FileNotFoundError("BBBC039 images missing")
    return fs


def bbbc002_paths():
    fs = sorted(glob.glob(str(DATA / "drosophila_kc167_1_images" / "*.TIF")))
    if not fs:
        raise FileNotFoundError("BBBC002 images missing")
    return fs


def load_fov(path, bit_depth):
    x = io.imread(path).astype(np.float32) / float((1 << bit_depth) - 1)
    return x


def split_indices(n, seed=0, n_val=20, n_test=40):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test = np.sort(perm[:n_test])
    val = np.sort(perm[n_test:n_test + n_val])
    train = np.sort(perm[n_test + n_val:])
    return dict(train=train.tolist(), val=val.tolist(), test=test.tolist())


def _dedupe(paths):
    """Drop duplicate extractions of the same file by basename."""
    seen = {}
    for f in sorted(paths):
        seen.setdefault(Path(f).name, f)
    return list(seen.values())


def _glob(pattern):
    return sorted(glob.glob(str(DATA / pattern), recursive=True))


def bbbc014_paths(channel=1):
    fs = _glob("BBBC014_v1_images/**/*.Bmp")
    return _dedupe(f for f in fs
                   if Path(f).name.startswith(f"Channel {channel}-"))


def bbbc015_paths():
    # Files end in _<field>_<channel>; use the first channel only.
    fs = _dedupe(_glob("BBBC015_v1_images/**/*.JPG"))
    return [f for f in fs if f.lower().endswith("_0.jpg")]


def bbbc016_paths(channel=0):
    fs = _glob("BBBC016_v1_images/**/*.TIF")
    return _dedupe(f for f in fs if re.search(rf"d{channel}\.TIF$", f, re.I))


def bbbc018_paths(channel="actin"):
    fs = _glob("BBBC018_v1_images/**/*.DIB")
    return _dedupe(f for f in fs
                   if f.lower().endswith(f"-{channel.lower()}.dib"))


DATASETS = {
    "bbbc014": dict(fn=bbbc014_paths, channel=1, scale=255.0),
    "bbbc015": dict(fn=bbbc015_paths, channel=None, scale=255.0),
    "bbbc016": dict(fn=bbbc016_paths, channel=0, scale=255.0),
    "bbbc018": dict(fn=bbbc018_paths, channel="actin", scale=255.0),
}


def dataset_paths(name, channel=None):
    spec = DATASETS[name]
    ch = spec["channel"] if channel is None else channel
    if ch is None:
        return spec["fn"](), spec["scale"]
    return spec["fn"](ch), spec["scale"]


def load_image(path, scale):
    x = io.imread(path)
    if x.ndim == 3:
        if x.shape[2] == 4:
            x = x[..., :3]
        if path.lower().endswith("-actin.dib"):
            # BBBC018 actin is encoded in the green channel; red is empty
            # and blue contains a different stain.
            x = x[..., 1].astype(np.float32) / float(scale)
        else:
            x = rgb2gray(x).astype(np.float32)
    else:
        x = x.astype(np.float32) / float(scale)
    return x


def split_paths(paths, seed=0, n_val=16, n_test=16):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))
    test = [paths[i] for i in np.sort(perm[:n_test])]
    val = [paths[i] for i in np.sort(perm[n_test:n_test + n_val])]
    train = [paths[i] for i in np.sort(perm[n_test + n_val:])]
    return dict(train=train, val=val, test=test)


def bbbc039_official_split():
    """The official BBBC039 100/50/50 FOV split (metadata/*.txt)."""
    base = DATA / "meta_039" / "metadata"
    by_stem = {Path(p).stem: p for p in bbbc039_paths()}
    out = {}
    for key, fn in [("train", "training.txt"), ("val", "validation.txt"),
                    ("test", "test.txt")]:
        lines = [x.strip() for x in (base / fn).read_text().splitlines()
                 if x.strip() and not x.startswith("#")]
        stems = [Path(x).stem for x in lines]
        out[key] = [by_stem[s] for s in stems]
    return out


def gaussian_psf(sigma=2.0, n=15):
    x = np.arange(n, dtype=np.float64) - n // 2
    xx, yy = np.meshgrid(x, x)
    p = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return (p / p.sum()).astype(np.float32)


def blur(x, sigma=2.0, n=15):
    return convolve(x.astype(np.float32), gaussian_psf(sigma, n), mode="reflect")


def airy_psf(n=31, na=1.40, lam=0.52, pix=0.10):
    """In-focus scalar Airy PSF for a circular aperture."""
    from scipy.special import j1
    x = (np.arange(n) - n // 2) * pix
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx * xx + yy * yy)
    v = 2.0 * np.pi * na * r / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (2.0 * j1(v) / v) ** 2
    p[v == 0] = 1.0
    return (p / p.sum()).astype(np.float32)


def blur_with(x, psf):
    return convolve(x.astype(np.float32), psf, mode="reflect")


def degrade(x, sigma=2.0, photons=300.0, rng=None, psf=None):
    """Poisson-corrupted blurred image; photons = full-scale photons/pixel."""
    rng = np.random.default_rng() if rng is None else rng
    y = blur_with(x, psf) if psf is not None else blur(x, sigma)
    return (rng.poisson(np.clip(y, 0, None) * photons) / photons).astype(np.float32)


def patches(img, size=128):
    h, w = img.shape
    out = []
    for i in range(0, h - size + 1, size):
        for j in range(0, w - size + 1, size):
            out.append((i, j, img[i:i + size, j:j + size]))
    return out


def psnr(gt, pred):
    return float(psnr_fn(gt, np.clip(pred, 0, 1), data_range=1.0))


def ssim(gt, pred):
    return float(ssim_fn(gt, np.clip(pred, 0, 1), data_range=1.0))


def fg_psnr(gt, pred, q=0.90):
    thr = np.quantile(gt, q)
    m = gt >= thr
    if m.sum() < 16:
        return float("nan")
    d = (gt[m] - np.clip(pred, 0, 1)[m]) ** 2
    mse = float(d.mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
