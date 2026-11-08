"""Generate real-data LaTeX macros and table from result JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path("results")
PAPER = Path("paper")
PAPER.mkdir(exist_ok=True)


def mean(rows, key, j=0):
    return float(np.mean([r[key][j] for r in rows]))


D = json.load(open(RES / "real_small_test_all.json"))
A_PATH = RES / "real_affine_test_all_strong.json"
if not A_PATH.exists():
    A_PATH = RES / "real_affine_test_all.json"
A = json.load(open(A_PATH))
rows = D["rows"]
aff = A["rows"]
tuned = D["tuned"]

from scipy.stats import wilcoxon

macros = []
macros.append(r"\newcommand{\RealN}{%d}" % len(aff))
macros.append(r"\newcommand{\RealNoisyP}{%.2f}" % mean(rows, "noisy", 0))
macros.append(r"\newcommand{\RealNoisyS}{%.3f}" % mean(rows, "noisy", 1))
for key, tag in [("bm3d", "BMThree"), ("nlm", "Nlm"),
                 ("tv", "Tv"), ("unet", "Unet")]:
    macros.append(r"\newcommand{\Real%sP}{%.2f}"
                  % (tag, mean(aff, key, 0)))
    macros.append(r"\newcommand{\Real%sS}{%.3f}"
                  % (tag, mean(aff, key, 1)))
for key, tag in [("bm3d", "BMThree"), ("nlm", "Nlm"), ("tv", "Tv")]:
    macros.append(r"\newcommand{\Real%sRawP}{%.2f}"
                  % (tag, mean(rows, key, 0)))
    macros.append(r"\newcommand{\Real%sRawS}{%.3f}"
                  % (tag, mean(rows, key, 1)))

d_b = np.array([r["unet"][0] - r["bm3d"][0] for r in aff])
d_n = np.array([r["unet"][0] - r["nlm"][0] for r in aff])
d_t = np.array([r["unet"][0] - r["tv"][0] for r in aff])
macros.append(r"\newcommand{\RealGain}{%.2f}" % d_b.mean())
macros.append(r"\newcommand{\RealGainRaw}{%.2f}"
              % (mean(aff, "unet", 0) - mean(rows, "bm3d", 0)))
rng = np.random.default_rng(0)
bs = [float(np.mean(rng.choice(d_b, len(d_b), replace=True)))
      for _ in range(5000)]
macros.append(r"\newcommand{\RealCI}{%.2f--%.2f}"
              % (np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
macros.append(r"\newcommand{\RealGainNlm}{%.2f}" % d_n.mean())
macros.append(r"\newcommand{\RealGainTv}{%.2f}" % d_t.mean())
macros.append(r"\newcommand{\RealWins}{%d}" % int((d_b > 0).sum()))
macros.append(r"\newcommand{\RealPvalue}{%.1e}" % float(wilcoxon(d_b).pvalue))
macros.append(r"\newcommand{\RealBmSigma}{%.1f}" % tuned["bm3d"])
macros.append(r"\newcommand{\RealNlmH}{%.1f}" % tuned["nlm"])
macros.append(r"\newcommand{\RealTvW}{%.2f}" % tuned["tv"])

# second supervised deep baseline (DnCNN)
try:
    DN_P = RES / "real_dncnn_strong.json"
    if not DN_P.exists():
        DN_P = RES / "real_dncnn_all.json"
    DN = json.load(open(DN_P))
    dn = np.array([r["dncnn"][0] for r in DN["rows"]])
    dn_s = np.array([r["dncnn"][1] for r in DN["rows"]])
    bm = np.array([r["bm3d"][0] for r in aff])
    un = np.array([r["unet"][0] for r in aff])
    macros.append(r"\newcommand{\RealDncnnP}{%.2f}" % dn.mean())
    macros.append(r"\newcommand{\RealDncnnS}{%.3f}" % dn_s.mean())
    macros.append(r"\newcommand{\RealDncnnGain}{%.2f}" % (dn - bm).mean())
    macros.append(r"\newcommand{\RealUnetGainDncnn}{%.2f}" % (un - dn).mean())
    try:
        UV = json.load(open(RES / "real_unet_vs_dncnn.json"))
        macros.append(r"\newcommand{\RealUnetDncnnCI}{%s--%s}" %
                      (f"{UV['ci'][0]:.2f}", f"{UV['ci'][1]:.2f}"))
        macros.append(r"\newcommand{\RealUnetDncnnWins}{%s}" %
                      UV["wins"])
    except FileNotFoundError:
        pass
except FileNotFoundError:
    pass

names = {"noisy": "Noisy", "bm3d": "BMThree", "nlm": "Nlm",
         "tv": "Tv", "unet": "Unet"}

# large-field domain-shift check (negative)
try:
    L_PATH = RES / "real_large_fair_strong.json"
    if not L_PATH.exists():
        L_PATH = RES / "real_large_fair.json"
    L = json.load(open(L_PATH))
    m = {k: float(np.mean([r[k][0] for r in L["rows"]])) for k in L["rows"][0]}
    macros.append(r"\newcommand{\LargeRealN}{%d}" % len(L["rows"]))
    for key, tag in names.items():
        macros.append(r"\newcommand{\LargeReal%sP}{%.2f}" % (tag, m[key]))
except FileNotFoundError:
    pass

(PAPER / "real_numbers.tex").write_text("\n".join(macros) + "\n")

tab = r"""\begin{tabular}{lcc}
\toprule
Method & PSNR (dB) & SSIM \\
\midrule
\multicolumn{3}{l}{\emph{AI4Life nuclei, \RealN\ held-out fields}} \\
Noisy input & %s & %s \\
TV denoising & %s & %s \\
NLM & %s & %s \\
BM3D & %s & %s \\
DnCNN (supervised) & \RealDncnnP & \RealDncnnS \\
U-Net & %s & %s \\
\bottomrule
\end{tabular}
""" % (r"\RealNoisyP", r"\RealNoisyS", r"\RealTvP", r"\RealTvS",
       r"\RealNlmP", r"\RealNlmS", r"\RealBMThreeP", r"\RealBMThreeS",
       r"\RealUnetP", r"\RealUnetS")
(PAPER / "tables").mkdir(exist_ok=True)
(PAPER / "tables" / "real.tex").write_text(tab)
print("wrote paper/real_numbers.tex and paper/tables/real.tex")
