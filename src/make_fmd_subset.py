"""Pre-specify a 150-field FMD test subset, stratified by acquisition group."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fmd_common import pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prereg/fmd_subset150.json")
    a = ap.parse_args()
    test = pairs("test")
    groups = defaultdict(list)
    for p in test:
        groups[tuple(p[2].split("_")[:3])].append(p)
    keys = sorted(groups)
    selected = []
    for k in keys:
        selected += sorted(groups[k], key=lambda p: p[2])[:4]
    for k in keys[:6]:
        extra = sorted(groups[k], key=lambda p: p[2])[4:5]
        selected += extra
    stems = sorted(p[2] for p in selected)
    assert len(stems) == 150, len(stems)
    out = dict(
        rule=("within each microscope/channel/level group, take the first four "
              "test fields in lexicographic order; add the fifth field from "
              "the first six groups in lexicographic order"),
        n=len(stems),
        groups=len(keys),
        stems=stems,
        group_counts={"_".join(k): len(v) for k, v in sorted(groups.items())},
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("selected", len(stems), "from", len(test), flush=True)


if __name__ == "__main__":
    main()
