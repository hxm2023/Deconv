"""Re-split the extracted FMD fields stratified by microscope/channel/level."""
from __future__ import annotations

import collections
import json

p = "datasets/fmd/split.json"
d = json.load(open(p))
allrec = d["train"] + d["val"] + d["test"]
groups = collections.defaultdict(list)
for r in allrec:
    groups[tuple(r["key"][:3])].append(r)

new = {"train": [], "val": [], "test": []}
for k in sorted(groups):
    rs = sorted(groups[k], key=lambda r: (int(r["key"][3]), int(r["frame"])))
    n = len(rs)
    nv = max(1, int(round(0.15 * n)))
    nt = max(1, int(round(0.15 * n)))
    new["train"] += rs[:n - nv - nt]
    new["val"] += rs[n - nv - nt:n - nt]
    new["test"] += rs[n - nt:]
json.dump(new, open(p, "w"), indent=2)
print({k: len(v) for k, v in new.items()})
print("groups", len(groups),
      "test groups", len({tuple(r["key"][:3]) for r in new["test"]}))
