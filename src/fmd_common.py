"""AI4Life-MDC25-FMD field-split subset."""
from __future__ import annotations

import json
from pathlib import Path

from skimage import io

from real_common import norm99

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "fmd"


def pairs(split):
    rec = json.load(open(DATA / "split.json"))[split]
    out = []
    for r in rec:
        out.append((str(DATA / r["noisy"]), str(DATA / r["gt"]),
                    Path(r["gt"]).stem))
    return out


def load_pair(npath, gpath):
    n = io.imread(npath).astype("float32")
    g = io.imread(gpath).astype("float32")
    return norm99(n), norm99(g)
