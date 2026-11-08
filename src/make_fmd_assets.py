"""Generate FMD real-data macros and table from results/fmd_test.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path("results")
PAPER = Path("paper")
D_PATH = RES / "fmd_test150_micro.json"
if not D_PATH.exists():
    D_PATH = RES / "fmd_test150.json"
if not D_PATH.exists():
    D_PATH = RES / "fmd_test.json"
D = json.load(open(D_PATH))
rows = D["rows"]

macros = [r"\newcommand{\FmdN}{%d}" % len(rows)]
names = {"noisy": "Noisy", "bm3d": "BMThree", "nlm": "Nlm",
         "tv": "Tv", "unet": "Unet"}
for key, tag in names.items():
    macros.append(r"\newcommand{\Fmd%sP}{%.2f}"
                  % (tag, float(np.mean([r[key][0] for r in rows]))))
    macros.append(r"\newcommand{\Fmd%sS}{%.3f}"
                  % (tag, float(np.mean([r[key][1] for r in rows]))))
d = np.array([r["unet"][0] - r["bm3d"][0] for r in rows])
rng = np.random.default_rng(0)
bs = [float(np.mean(rng.choice(d, len(d), replace=True)))
      for _ in range(5000)]
S = dict(n=len(d), mean=float(d.mean()),
         ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
         wins=int((d > 0).sum()), p=float(D["paired"]["bm3d"]["p"]))
json.dump(S, open(RES / "fmd_stats.json", "w"), indent=2)
macros.append(r"\newcommand{\FmdGain}{%.2f}" % d.mean())
macros.append(r"\newcommand{\FmdWins}{%d}" % int((d > 0).sum()))
macros.append(r"\newcommand{\FmdPvalue}{%.1e}" % D["paired"]["bm3d"]["p"])
macros.append(r"\newcommand{\FmdCI}{%.2f--%.2f}" % (S["ci"][0], S["ci"][1]))
try:
    ST = json.load(open(RES / "fmd_strata150_micro.json"))
    macros.append(r"\newcommand{\FmdStrataMin}{%.1f}" % ST["mean_range"][0])
    macros.append(r"\newcommand{\FmdStrataMax}{%.1f}" % ST["mean_range"][1])
    macros.append(r"\newcommand{\FmdStrataN}{%d}" % len(ST["groups"]))
except FileNotFoundError:
    pass
(PAPER / "fmd_numbers.tex").write_text("\n".join(macros) + "\n")
print("wrote paper/fmd_numbers.tex")
