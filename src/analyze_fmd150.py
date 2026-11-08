"""Per-stratum summary of the pre-specified 150-field FMD subset."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fmd_common import pairs


def main():
    keep = set(json.load(open("prereg/fmd_subset150.json"))["stems"])
    test = [p for p in pairs("test") if p[2] in keep]
    import sys
    D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/fmd_test150.json"))
    rows = D["rows"]
    assert len(test) == len(rows), (len(test), len(rows))
    groups = defaultdict(list)
    for p, r in zip(test, rows):
        g = "_".join(p[2].split("_")[:3])
        groups[g].append(r["unet"][0] - r["bm3d"][0])
    out = {}
    for g, d in sorted(groups.items()):
        out[g] = dict(n=len(d), mean=float(np.mean(d)),
                      min=float(np.min(d)), max=float(np.max(d)),
                      wins=int(sum(x > 0 for x in d)))
    vals = [v["mean"] for v in out.values()]
    print("groups", len(out), "mean range", round(min(vals), 2),
          round(max(vals), 2), flush=True)
    for g, v in out.items():
        print(g, v, flush=True)
    json.dump(dict(groups=out, mean_range=[min(vals), max(vals)]),
              open("results/fmd_strata150_micro.json", "w"), indent=2)


if __name__ == "__main__":
    main()
