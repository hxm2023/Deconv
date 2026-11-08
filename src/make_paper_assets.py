"""Generate paper/numbers.tex and tables from result JSON files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAPER = Path("paper")
TABLES = PAPER / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

MAIN = json.load(open("results/bench_summary.json"))
PROTO = json.load(open("results/protocol_dose.json"))
EXT = json.load(open("results/bench_external_d30.json"))
UNC = json.load(open("results/uncertainty_fov.json"))
SCR = json.load(open("results/scr_eval.json"))
DOSES = [3, 10, 30, 100, 300]
METHODS = ["tv", "nlm", "wiener", "rl", "unet"]
LABEL = dict(tv="TV", nlm="NLM", wiener="Wiener", rl="RL", unet="U-Net")
MET = ["psnr", "ssim", "fg"]

lines = []
def macro(name, val):
    lines.append(rf"\newcommand{{\{name}}}{{{val}}}")

def main_mean(m, d):
    return np.asarray(MAIN["fov"][m][str(d)]).mean(0)

def eval_mean(path, dose):
    j = json.load(open(path))
    rows = j["test"][str(float(dose))]
    a = np.array([[r["fov_mean_psnr"], r["fov_mean_ssim"], r["fov_mean_fg"]]
                  for r in rows])
    return a.mean(0)

def bench_mean(path):
    j = json.load(open(path))
    rec = j["main"]["records"]
    fovs = sorted(set(r["fov"] for r in rec))
    out = {}
    for m in ["tv", "nlm", "wiener", "rl", "unet"]:
        rows = []
        for f in fovs:
            sel = [r for r in rec if r["fov"] == f]
            rows.append([np.mean([r[m][k] for r in sel]) for k in MET])
        out[m] = np.array(rows).mean(0)
    return out

def roc_auc(scores, labels):
    order = np.argsort(-np.asarray(scores))
    lab = np.asarray(labels)[order].astype(float)
    tpr = np.r_[0, np.cumsum(lab) / max(lab.sum(), 1)]
    fpr = np.r_[0, np.cumsum(1 - lab) / max((1 - lab).sum(), 1)]
    return float(np.trapezoid(tpr, fpr))

# ---- numbers.tex ----
for m, tag in [("unet", "UNet"), ("tv", "TV"), ("nlm", "Nlm")]:
    for d, dt in [(3, "three"), (30, "thirty"), (300, "threehundred")]:
        v = main_mean(m, d)
        macro(f"{tag}P{dt}", f"{v[0]:.2f}")
        macro(f"{tag}S{dt}", f"{v[1]:.3f}")
        macro(f"{tag}F{dt}", f"{v[2]:.2f}")

# protocol values
def proto_val(metric, target, method):
    for row in PROTO[metric]:
        if abs(row["target"] - target) < 1e-9:
            return row["dose"].get(method)
    return None

macro("DoseFactorLow", "3")
macro("DoseFactorHigh", "10")
macro("DosePSNRThirtyTwo", "10")
for name, target in [("PSNRThirty", 30), ("PSNRThirtyTwo", 32),
                     ("PSNRThirtyFour", 34), ("SSIMNine", 0.90),
                     ("SSIMNineTwo", 0.92), ("FGTwoSix", 26)]:
    metric = "psnr" if name.startswith("PSNR") else (
        "ssim" if name.startswith("SSIM") else "fg")
    macro(f"Dose{name}TV", str(proto_val(metric, target, "tv") or "--"))
    macro(f"Dose{name}U", str(proto_val(metric, target, "unet") or "--"))
    macro(f"Dose{name}Nlm", str(proto_val(metric, target, "nlm") or "--"))

# specialist ablation
s3 = eval_mean("results/spec3_eval.json", 3)
s30 = eval_mean("results/spec30_eval.json", 30)
s100 = eval_mean("results/spec100_eval.json", 100)
c3 = eval_mean("results/unet_dose_response.json", 3)
c30 = eval_mean("results/unet_dose_response.json", 30)
c100 = eval_mean("results/unet_dose_response.json", 100)
macro("SpecPthree", f"{s3[0]:.2f}")
macro("CondPthree", f"{c3[0]:.2f}")
macro("SpecPthirty", f"{s30[0]:.2f}")
macro("CondPthirty", f"{c30[0]:.2f}")
macro("SpecPonehundred", f"{s100[0]:.2f}")
macro("CondPonehundred", f"{c100[0]:.2f}")

# cross dataset
def ext_delta(key, base):
    rec = EXT[key]["records"]
    fovs = sorted(set(r["fov"] for r in rec))
    vals = {m: [] for m in [base, "unet"]}
    for f in fovs:
        sel = [r for r in rec if r["fov"] == f]
        for m in vals:
            vals[m].append(np.mean([r[m]["psnr"] for r in sel]))
    return float(np.mean(np.asarray(vals["unet"]) - np.asarray(vals[base])))
d14 = ext_delta("main", "tv")
d15 = ext_delta("bbbc015", "tv")
d16 = ext_delta("bbbc016", "tv")
d18 = ext_delta("bbbc018", "tv")
macro("CrossBbbcFourteen", f"{d14:.2f}")
macro("CrossBbbcFifteen", f"{d15:.2f}")
macro("CrossBbbcSixteen", f"{d16:.2f}")
macro("CrossBbbcEighteen", f"{d18:.2f}")
macro("CrossMin", f"{min(d14, d15, d16, d18):.1f}")
macro("CrossMax", f"{max(d14, d15, d16, d18):.1f}")

# uncertainty
def auc_for(d):
    scores = [r[str(d)]["unc"] for r in UNC]
    labels = [r[str(d)]["psnr"] < 34 for r in UNC]
    return roc_auc(scores, labels)
macro("AucThree", f"{auc_for(3):.2f}")
macro("AucTen", f"{auc_for(10):.2f}")
macro("AucThirty", f"{auc_for(30):.2f}")
from scipy.stats import spearmanr
rho = spearmanr([r["3"]["unc"] for r in UNC], [r["3"]["err"] for r in UNC]).statistic
macro("RhoThree", f"{rho:.2f}")

# scrambled
scr3 = eval_mean("results/scr_eval.json", 3)
scr30 = eval_mean("results/scr_eval.json", 30)
macro("ScrPthree", f"{scr3[0]:.2f}")
macro("ScrPthirty", f"{scr30[0]:.2f}")

# sigma-conditioned
sig = {}
for s in [1.5, 2.0, 2.5]:
    f = f"results/bench_sigma{str(s).replace('.', '')}_d30.json"
    try:
        sig[s] = bench_mean(f)
    except Exception:
        sig[s] = None
for s, tag in [(1.5, "fifteen"), (2.0, "two"), (2.5, "twopfive")]:
    if sig.get(s) is not None:
        macro(f"SigmaU{tag}", f"{sig[s]['unet'][0]:.2f}")
        macro(f"SigmaTV{tag}", f"{sig[s]['tv'][0]:.2f}")
        macro(f"SigmaNlm{tag}", f"{sig[s]['nlm'][0]:.2f}")
    else:
        for pre in ["SigmaU", "SigmaTV", "SigmaNlm"]:
            macro(f"{pre}{tag}", "--")

(PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")

# ---- tables/perf.tex ----
out = [r"\begin{tabular}{lccccc}", r"\toprule",
       r" & TV & NLM & Wiener & RL & U-Net \\", r"\midrule"]
for d in [3, 30, 300]:
    out.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{d} photons per pixel}}}} \\")
    for mi, mname in [(0, "PSNR (dB)"), (1, "SSIM"), (2, "Foreground PSNR (dB)")]:
        vals = []
        for m in METHODS:
            v = main_mean(m, d)[mi]
            vals.append(f"{v:.2f}" if mi != 1 else f"{v:.3f}")
        out.append(f"{mname} & " + " & ".join(vals) + r" \\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule"
out.append(r"\end{tabular}")
(TABLES / "perf.tex").write_text("\n".join(out) + "\n")

# ---- tables/protocol.tex ----
out = [r"\begin{tabular}{llccccc}", r"\toprule",
       r"Target & Metric & TV & NLM & Wiener & RL & U-Net \\", r"\midrule"]
for metric, label, targets in [
        ("psnr", "PSNR", [30, 32, 34]),
        ("ssim", "SSIM", [0.85, 0.90, 0.92]),
        ("fg", "Foreground PSNR", [22, 24, 26])]:
    for T in targets:
        cells = []
        for m in ["tv", "nlm", "wiener", "rl", "unet"]:
            v = proto_val(metric, T, m)
            cells.append("--" if v is None else str(v))
        lab = f"{T:g}" + (" dB" if metric != "ssim" else "")
        out.append(f"{label} & {lab} & " + " & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "protocol.tex").write_text("\n".join(out) + "\n")

# ---- tables/cross.tex ----
out = [r"\begin{tabular}{lccccc}", r"\toprule",
       r"Dataset & TV & NLM & U-Net & $\Delta$ vs TV & $\Delta$ vs NLM \\",
       r"\midrule"]
for key, name in [("main", "BBBC014 (train)"), ("bbbc015", "BBBC015"),
                  ("bbbc016", "BBBC016"), ("bbbc018", "BBBC018")]:
    rec = EXT[key]["records"]
    fovs = sorted(set(r["fov"] for r in rec))
    arr = {}
    for m in ["tv", "nlm", "unet"]:
        vals = []
        for f in fovs:
            sel = [r for r in rec if r["fov"] == f]
            vals.append(np.mean([r[m]["psnr"] for r in sel]))
        arr[m] = np.asarray(vals)
    out.append(f"{name} & {arr['tv'].mean():.2f} & {arr['nlm'].mean():.2f} & "
               f"{arr['unet'].mean():.2f} & "
               f"{arr['unet'].mean()-arr['tv'].mean():+.2f} & "
               f"{arr['unet'].mean()-arr['nlm'].mean():+.2f} \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "cross.tex").write_text("\n".join(out) + "\n")

# ---- tables/ablation.tex ----
out = [r"\begin{tabular}{lcccc}", r"\toprule",
       r"Dose & Specialist & Dose-cond. & Scrambled & TV \\", r"\midrule"]
for d, s, c, sc in [(3, s3, c3, scr3), (30, s30, c30, scr30),
                    (100, s100, c100, eval_mean("results/scr_eval.json", 100))]:
    out.append(f"{d} photons & {s[0]:.2f} & {c[0]:.2f} & {sc[0]:.2f} & "
               f"{main_mean('tv', d)[0]:.2f} \\\\")
out += [r"\midrule",
        r"\multicolumn{5}{l}{\emph{Dose+PSF model, 30 photons}} \\",
        r"$\sigma$ (pixels) & U-Net & TV & NLM & \\", r"\midrule"]
for s, tag in [(1.5, "fifteen"), (2.0, "two"), (2.5, "twopfive")]:
    if sig.get(s) is not None:
        out.append(f"{s:g} & {sig[s]['unet'][0]:.2f} & {sig[s]['tv'][0]:.2f} & "
                   f"{sig[s]['nlm'][0]:.2f} & \\\\")
    else:
        out.append(f"{s:g} & -- & -- & -- & \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "ablation.tex").write_text("\n".join(out) + "\n")

print("wrote paper/numbers.tex and tables")
