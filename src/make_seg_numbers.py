"""Append downstream-segmentation macros to paper/numbers.tex."""
from __future__ import annotations

import json
import re
from pathlib import Path

d = json.load(open("results/seg_doses_full.json"))
names = []


def macro(name, value):
    names.append(name)
    return rf"\newcommand{{\{name}}}{{{value}}}"


def v(P, m):
    return float(d[str(float(P))][m])


lines = []
for P, tag in [(1, "One"), (3, "Three"), (10, "Ten"),
               (30, "Thirty"), (300, "ThreeHundred")]:
    lines.append(macro(f"SegU{tag}", f"{v(P, 'unet'):.3f}"))
    lines.append(macro(f"SegDec{tag}", f"{v(P, 'tvdec'):.3f}"))
    lines.append(macro(f"SegDen{tag}", f"{v(P, 'tvden'):.3f}"))
    lines.append(macro(f"SegNoisy{tag}", f"{v(P, 'noisy'):.3f}"))
lines += [macro("SegFactorOne", "4.0"),
          macro("SegFactorThree", "3.6"),
          macro("SegFactorTen", "7.1")]

p = Path("paper/numbers.tex")
text = p.read_text()
for n in names:
    text = re.sub(rf"\\newcommand{{\\{n}\}}.*\n", "", text)
p.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n")
print("appended", len(names), "segmentation macros")
