"""BBBC039 nuclei segmentation + low-dose deconvolution data."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from skimage import io

from common import DATA, bbbc039_official_split, degrade, load_image, patches

MASK_DIR = DATA / "masks_039" / "masks"


def mask_for(image_path):
    stem = Path(image_path).stem
    p = MASK_DIR / f"{stem}.png"
    m = io.imread(p)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.float32)


def load_pair(image_path, scale=4095.0):
    return load_image(image_path, scale), mask_for(image_path)


def random_pair_patch(img, mask, size, rng):
    h, w = img.shape
    i = int(rng.integers(0, h - size + 1))
    j = int(rng.integers(0, w - size + 1))
    return img[i:i + size, j:j + size], mask[i:i + size, j:j + size]


def eval_patches(img, mask, size=128, n=10):
    ps = list(patches(img, size))
    step = max(1, len(ps) // n)
    return [(i, j, img[i:i + size, j:j + size],
             mask[i:i + size, j:j + size]) for i, j, _ in ps[::step][:n]]


def dice(a, b):
    a = a > 0
    b = b > 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def iou(a, b):
    a = a > 0
    b = b > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def psnr(gt, pred):
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(gt, np.clip(pred, 0, 1),
                                         data_range=1.0))


def ssim(gt, pred):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(gt, np.clip(pred, 0, 1),
                                       data_range=1.0))
