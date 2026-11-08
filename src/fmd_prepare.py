"""Extract a field-split subset of AI4Life-MDC25-FMD.

The archive contains gt/ and noisy/ 256x256 TIFFs named
MICROSCOPE_CHANNEL_LLEVEL_FIELD_FRAME.tif. We select one frame per field,
stratify by microscope, and split fields 70/15/15 to avoid frame leakage.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"(gt|noisy)/([A-Z]+)_([BGR])_L([0-9])_([0-9]+)_([0-9]+)\.tif$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="datasets/ai4life_fmd/Archive.zip")
    ap.add_argument("--out", default="datasets/fmd")
    ap.add_argument("--fields", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    z = zipfile.ZipFile(a.zip)
    names = set(z.namelist())
    by_field = defaultdict(list)
    for n in names:
        m = PAT.match(n)
        if not m or m.group(1) != "gt":
            continue
        mic, ch, lev, field, frame = m.groups()[1:]
        key = (mic, ch, lev, field)
        noisy = n.replace("gt/", "noisy/", 1)
        if noisy in names:
            by_field[key].append((int(frame), n, noisy))

    rng = np.random.default_rng(a.seed)
    keys = list(by_field)
    per_mic = defaultdict(list)
    for k in keys:
        per_mic[k[0]].append(k)
    sel = []
    n_per = a.fields // len(per_mic)
    for mic, ks in sorted(per_mic.items()):
        ks = sorted(ks)
        take = min(n_per, len(ks))
        idx = rng.choice(len(ks), size=take, replace=False)
        sel.extend(ks[i] for i in idx)
    rng.shuffle(sel)

    split = {"train": [], "val": [], "test": []}
    # stratify by microscope within each split
    by_mic = defaultdict(list)
    for k in sel:
        by_mic[k[0]].append(k)
    for mic, ks in sorted(by_mic.items()):
        ks = sorted(ks)
        n = len(ks)
        n_val = max(1, int(round(0.15 * n)))
        n_test = max(1, int(round(0.15 * n)))
        split["train"].extend(ks[:n - n_val - n_test])
        split["val"].extend(ks[n - n_val - n_test:n - n_test])
        split["test"].extend(ks[n - n_test:])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = {"train": [], "val": [], "test": []}
    for sname, ks in split.items():
        for k in ks:
            frame, gt, noisy = sorted(by_field[k])[0]
            z.extract(gt, out)
            z.extract(noisy, out)
            records[sname].append(dict(key=list(k), frame=frame,
                                       gt=gt, noisy=noisy))
    (out / "split.json").write_text(json.dumps(records, indent=2))
    print({k: len(v) for k, v in records.items()}, flush=True)
    print("wrote", out / "split.json")


if __name__ == "__main__":
    main()
