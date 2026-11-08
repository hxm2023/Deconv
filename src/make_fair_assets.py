"""Generate fair-baseline paper assets from result JSON files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAPER = Path("paper")
TABLES = PAPER / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
D = json.load(open("results/tvdeconv_doses.json"))
BLINDFAIR = json.load(open("results/blind_fair_doses.json"))
BLINDEXT = json.load(open("results/blind_external30.json"))
SCR2 = json.load(open("results/scr2_matched.json"))
BLINDSIG = json.load(open("results/blind_sigma.json"))
DOSES = [3, 10, 30, 100, 300]
NOISY = json.load(open("results/noisy_baseline.json"))
EXT = json.load(open("results/tvdeconv_external30.json"))
try:
    EXT2 = json.load(open("results/external_tuned.json"))
except FileNotFoundError:
    EXT2 = None
BLIND = json.load(open("results/blind2_eval.json"))
SCR = json.load(open("results/scr_eval.json"))
try:
    SCRSTRONG = json.load(open("results/scr_blind_strong.json"))
except FileNotFoundError:
    SCRSTRONG = None
try:
    SPECSTRONG = json.load(open("results/spec_strong.json"))
except FileNotFoundError:
    SPECSTRONG = None
SIGU = json.load(open("results/dcond_mismatch.json"))
SIGD = json.load(open("results/tvdeconv_sigma.json"))
UNC = json.load(open("results/uncertainty_fov.json"))

lines = []
def macro(name, val):
    lines.append(rf"\newcommand{{\{name}}}{{{val}}}")

def fov(dose, key):
    return np.array([np.mean(r[key]) for r in D[str(dose)]["rows"]])

def mean(dose, key):
    return float(fov(dose, key).mean())

def bmean(dose, key):
    return float(np.mean([r[key] for r in BLINDFAIR[str(dose)]]))

def bfov(dose, key):
    return np.array([r[key] for r in BLINDFAIR[str(dose)]])

for dose, tag in [(3, "three"), (10, "ten"), (30, "thirty"),
                  (100, "onehundred"), (300, "threehundred")]:
    n = NOISY[str(dose)]
    macro(f"NoisyP{tag}", f"{n[0]:.2f}")
    macro(f"NoisyS{tag}", f"{n[1]:.3f}")
    macro(f"NoisyF{tag}", f"{n[2]:.2f}")
    for m, mt in [("tv_denoise", "Den"), ("tv_deconv", "Dec")]:
        macro(f"{mt}P{tag}", f"{mean(dose, m):.2f}")
        macro(f"{mt}S{tag}", f"{mean(dose, m + '_ssim'):.3f}")
        macro(f"{mt}F{tag}", f"{mean(dose, m + '_fg'):.2f}")
    macro(f"UNetP{tag}", f"{bmean(dose, 'psnr'):.2f}")
    macro(f"UNetS{tag}", f"{bmean(dose, 'ssim'):.3f}")
    macro(f"UNetF{tag}", f"{bmean(dose, 'fg'):.2f}")

def protocol(method, metric, T):
    if method == "unet":
        for dose in DOSES:
            if float((bfov(dose, metric) >= T).mean()) >= 0.9:
                return dose
        return None
    key = method if metric == "psnr" else method + "_" + metric
    for dose in DOSES:
        if float((fov(dose, key) >= T).mean()) >= 0.9:
            return dose
    return None

for dose, tag in [(3, "Three"), (10, "Ten"), (30, "Thirty"),
                  (100, "Hundred"), (300, "ThreeHundred")]:
    u = bfov(dose, "psnr")
    b = fov(dose, "tv_deconv")
    d = u - b
    rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(10000)]
    macro(f"DecGain{tag}", f"{d.mean():.2f}")
    macro(f"DecCI{tag}",
          f"{np.percentile(bs, 2.5):.2f}--{np.percentile(bs, 97.5):.2f}")
macro("DenGainThree", f"{bmean(3, 'psnr') - mean(3, 'tv_denoise'):.2f}")
for dose, tag in [(3, "Three"), (30, "Thirty"), (300, "ThreeHundred")]:
    best = max(mean(dose, "tv_deconv"), mean(dose, "tv_denoise"))
    macro(f"BestGain{tag}", f"{bmean(dose, 'psnr') - best:.2f}")
_best_gains = [bmean(d, "psnr") - max(mean(d, "tv_deconv"),
                                      mean(d, "tv_denoise")) for d in DOSES]
macro("BestGainMin", f"{min(_best_gains):.2f}")
macro("BestGainMax", f"{max(_best_gains):.2f}")

for name, metric, T in [("PSNRThirtyTwo", "psnr", 32),
                        ("SSIMNine", "ssim", 0.90),
                        ("SSIMNineTwo", "ssim", 0.92),
                        ("FGTwoSix", "fg", 26)]:
    dec = protocol("tv_deconv", metric, T)
    un = protocol("unet", metric, T)
    macro(f"{name}Dec", str(dec) if dec else "--")
    macro(f"{name}U", str(un) if un else "--")
    macro(f"{name}Factor", f"{dec/un:.1f}" if dec and un else "--")

# cross-dataset
def emean(rows, key):
    return float(np.mean([r[key] for r in rows]))
macro("CrossDecFourteen", f"{bmean(30,'psnr')-mean(30,'tv_deconv'):.2f}")
for key, tag in [("bbbc015", "Fifteen"), ("bbbc016", "Sixteen"),
                 ("bbbc018", "Eighteen")]:
    if EXT2:
        macro(f"CrossDec{tag}", f"{EXT2[key]['delta_mean']:.2f}")
    else:
        macro(f"CrossDec{tag}",
              f"{np.mean(BLINDEXT[key])-emean(EXT[key],'tvdec'):.2f}")
if EXT2:
    _cd = [EXT2[k]["delta_mean"] for k in EXT2]
else:
    _cd = [np.mean(BLINDEXT[k])-emean(EXT[k],'tvdec') for k in BLINDEXT]
macro("CrossDecMin", f"{min(_cd):.2f}")
macro("CrossDecMax", f"{max(_cd):.2f}")

# matched ablation values
try:
    MATCH = json.load(open("results/matched_ablation.json"))
except Exception:
    MATCH = {}
for name, pre in [("specialist", "Spec"), ("dcond", "Cond")]:
    for dose, tag in [(3, "three"), (30, "thirty"), (100, "onehundred")]:
        if name == "specialist" and SPECSTRONG:
            v = SPECSTRONG.get(str(dose))
        else:
            v = MATCH.get(name, {}).get(str(dose), {})
        macro(f"{pre}P{tag}", "--" if not v else f"{v['psnr']:.2f}")
        macro(f"{pre}S{tag}", "--" if not v else f"{v['ssim']:.3f}")
        macro(f"{pre}F{tag}", "--" if not v else f"{v['fg']:.2f}")
for dose, tag in [(3, "three"), (30, "thirty"), (100, "onehundred")]:
    macro(f"BlindP{tag}", f"{bmean(dose, 'psnr'):.2f}")
    macro(f"BlindS{tag}", f"{bmean(dose, 'ssim'):.3f}")
    macro(f"BlindF{tag}", f"{bmean(dose, 'fg'):.2f}")
    if SCRSTRONG:
        macro(f"ScrP{tag}", f"{np.mean(SCRSTRONG[str(dose)]):.2f}")
    else:
        macro(f"ScrP{tag}", f"{np.mean(SCR2[str(dose)]):.2f}")
deltas = [MATCH["dcond"][str(d)]["psnr"] - bmean(d, "psnr")
          for d in [3, 30, 100]]
macro("CondDeltaLow", f"{min(deltas):.2f}")
macro("CondDeltaHigh", f"{max(deltas):.2f}")
CS = {}
try:
    CS = json.load(open("results/dcond_strong_doses.json"))
    for dose, tag in [(3, "three"), (30, "thirty"), (100, "onehundred")]:
        p = float(np.mean([r["psnr"] for r in CS[str(dose)]]))
        macro(f"CondStrongP{tag}", f"{p:.2f}")
    ds = [float(np.mean([r["psnr"] for r in CS[str(d)]])) - bmean(d, "psnr")
          for d in [3, 30, 100]]
    macro("CondStrongLow", f"{min(ds):+.2f}")
    macro("CondStrongHigh", f"{max(ds):+.2f}")
    macro("CondStrongDelta", f"{max(abs(x) for x in ds):.2f}")
except FileNotFoundError:
    pass

# uncertainty AUC (3, 10) and failure count at 30
def auc(d):
    order = np.argsort(-np.asarray([r[str(d)]["unc"] for r in UNC]))
    lab = np.asarray([r[str(d)]["psnr"] < 34 for r in UNC])[order].astype(float)
    tpr = np.r_[0, np.cumsum(lab) / max(lab.sum(), 1)]
    fpr = np.r_[0, np.cumsum(1 - lab) / max((1 - lab).sum(), 1)]
    return float(np.trapezoid(tpr, fpr))
macro("AucThree", f"{auc(3):.2f}")
macro("AucTen", f"{auc(10):.2f}")
macro("NFailThirty", str(int(sum(r["30"]["psnr"] < 34 for r in UNC))))
from scipy.stats import spearmanr
macro("RhoThree", f"{spearmanr([r['3']['unc'] for r in UNC], [r['3']['err'] for r in UNC]).statistic:.2f}")

# PSF robustness (dose-only); eval_physics_cond keys doses as floats
for sigma, tag in [("1.5", "fifteen"), ("2.0", "two"), ("2.5", "twopfive")]:
    macro(f"SigmaUP{tag}", f"{np.mean(BLINDSIG[sigma]):.2f}")
    macro(f"SigmaDec{tag}", f"{np.mean(SIGD[sigma]):.2f}")

# physical Airy PSF: Gaussian-trained model and PSF-mixed variant
try:
    AIRY0 = json.load(open("results/psf_airy.json"))
    for dose, tag in [("3", "Three"), ("30", "Thirty"), ("300", "ThreeHundred")]:
        rows = AIRY0["rows"][dose]
        u = np.mean([r["unet"] for r in rows])
        t = np.mean([r["tv"] for r in rows])
        macro(f"AiryStrongU{tag}", f"{u:.2f}")
        macro(f"AiryStrongGain{tag}", f"{u - t:+.2f}")
except FileNotFoundError:
    pass
try:
    AIRY = json.load(open("results/psf_airy_mix.json"))
    AIRYS = json.load(open("results/psf_airy_mix_stats.json"))
    for dose, tag in [("3", "Three"), ("30", "Thirty"), ("300", "ThreeHundred")]:
        rows = AIRY["rows"][dose]
        macro(f"AiryU{tag}", f"{np.mean([r['unet'] for r in rows]):.2f}")
        macro(f"AiryTv{tag}", f"{np.mean([r['tv'] for r in rows]):.2f}")
        macro(f"AiryGain{tag}", f"{AIRYS[dose]['mean']:.2f}")
        macro(f"AiryCI{tag}",
              f"{AIRYS[dose]['ci'][0]:.2f}--{AIRYS[dose]['ci'][1]:.2f}")
except FileNotFoundError:
    pass
try:
    PMG = json.load(open("results/blind_psfmix_gauss.json"))
    for dose, tag in [("3", "Three"), ("30", "Thirty"), ("300", "ThreeHundred")]:
        macro(f"PsfMixGaussP{tag}",
              f"{np.mean([r['psnr'] for r in PMG[dose]]):.2f}")
except FileNotFoundError:
    pass

# held-out doses (not used in training)
try:
    H = json.load(open("results/heldout_doses.json"))
    for mid, tag in [(5, "Five"), (50, "Fifty"), (200, "TwoHundred")]:
        macro(f"HeldoutP{tag}",
              f"{np.mean([r['psnr'] for r in H['rows'][str(mid)]]):.2f}")
    for c, tag in zip(H["interpolation"], ["Five", "Fifty", "TwoHundred"]):
        macro(f"HeldoutRes{tag}", f"{abs(c['residual']):.2f}")
except FileNotFoundError:
    pass

# speed
try:
    TIM = json.load(open("results/timing.json"))
    cpu = round(TIM["unet_cpu_ms"], 1)
    gpu = round(TIM["unet_gpu_ms"], 1)
    tv_ms = round(TIM["tvdec_cpu_ms"])
    macro("SpeedCPU", f"{cpu:.1f}")
    macro("SpeedGPU", f"{gpu:.1f}")
    macro("SpeedTV", f"{tv_ms:d}")
    macro("SpeedFactorCPU", f"{tv_ms / cpu:.1f}")
    macro("SpeedFactorGPU", f"{tv_ms / gpu:.0f}")
except FileNotFoundError:
    pass

(PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")

# ---- main table ----
out = [r"\begin{tabular}{lcccc}", r"\toprule",
       r" & Noisy & TV denoise & TV deconv & U-Net \\", r"\midrule"]
for dose in DOSES:
    out.append(rf"\multicolumn{{5}}{{l}}{{\emph{{$P={dose}$ (full-scale units)}}}} \\")
    for key, label, fmt in [("psnr", "PSNR (dB)", ".2f"),
                            ("ssim", "SSIM", ".3f"),
                            ("fg", "Foreground PSNR (dB)", ".2f")]:
        vals = []
        n = NOISY[str(dose)][{"psnr": 0, "ssim": 1, "fg": 2}[key]]
        vals.append(format(n, fmt))
        for m in ["tv_denoise", "tv_deconv"]:
            v = mean(dose, m + ("" if key == "psnr" else "_" + key))
            vals.append(format(v, fmt))
        vals.append(format(bmean(dose, key), fmt))
        out.append(f"{label} & " + " & ".join(vals) + r" \\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule"
out.append(r"\end{tabular}")
(TABLES / "perf.tex").write_text("\n".join(out) + "\n")

# ---- protocol table: validation-selected doses and their test success ----
try:
    PV = json.load(open("results/protocol_val.json"))["targets"]
    PV = {(r["metric"], float(r["target"])): r for r in PV}
except FileNotFoundError:
    PV = {}
out = [r"\begin{tabular}{llccccc}", r"\toprule",
       r"Metric & Target & TV deconv dose & U-Net dose & Factor & TV test & U-Net test \\",
       r"\midrule"]
for metric, label, targets in [("psnr", "PSNR", [30, 32, 34]),
                                ("ssim", "SSIM", [0.85, 0.90, 0.92]),
                                ("fg", "Foreground PSNR", [22, 24, 26])]:
    for T in targets:
        r = PV.get((metric, float(T)))
        lab = (f"{T:.2f}" if metric == "ssim" else f"{T:g}") + \
              (" dB" if metric != "ssim" else "")
        if not r:
            out.append(f"{label} & {lab} & -- & -- & -- & -- & -- \\\\")
            continue
        us = r["unet_test_success"]; ts = r["tv_test_success"]
        ok = (us is not None and ts is not None and us >= 0.9 and ts >= 0.9)
        fac = ("--" if not r["factor"] or not ok
               else f"{r['factor']:.1f}$\\times$")
        uc = "--" if us is None or us < 0.9 else f"{us*100:.0f}\\%"
        tc = "--" if ts is None or ts < 0.9 else f"{ts*100:.0f}\\%"
        out.append(f"{label} & {lab} & {r['tv_dose']} & {r['unet_dose']} "
                   f"& {fac} & {tc} & {uc} \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "protocol.tex").write_text("\n".join(out) + "\n")

# ---- test-oracle protocol table (upper bound) ----
out = [r"\begin{tabular}{llccc}", r"\toprule",
       r"Metric & Target & TV denoise & TV deconv & U-Net \\", r"\midrule"]
for metric, label, targets in [("psnr", "PSNR", [30, 32, 34]),
                                ("ssim", "SSIM", [0.85, 0.90, 0.92]),
                                ("fg", "Foreground PSNR", [22, 24, 26])]:
    for T in targets:
        cells = []
        for m in ["tv_denoise", "tv_deconv", "unet"]:
            v = protocol(m, metric, T)
            cells.append("--" if v is None else str(v))
        lab = (f"{T:.2f}" if metric == "ssim" else f"{T:g}") + \
              (" dB" if metric != "ssim" else "")
        out.append(f"{label} & {lab} & " + " & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "protocol_oracle.tex").write_text("\n".join(out) + "\n")

# ---- cross table ----
out = [r"\begin{tabular}{lccc}", r"\toprule",
       r"Dataset & TV deconv & U-Net & $\Delta$ [95\% CI] \\", r"\midrule"]
dmain = bfov(30, "psnr") - np.array(
    [np.mean(r["tv_deconv"]) for r in D["30"]["rows"]])
rows = [("BBBC014 (20 test FOVs)", mean(30, "tv_deconv"), bmean(30, "psnr"),
         float(dmain.mean()), dmain)]
for key, name in [("bbbc015", "BBBC015"), ("bbbc016", "BBBC016"),
                  ("bbbc018", "BBBC018")]:
    if EXT2:
        d = np.array(EXT2[key]["delta"])
        rows.append((f"{name} (15 fields)", EXT2[key]["tv_mean"],
                     EXT2[key]["unet_mean"], float(d.mean()), d))
    else:
        d = np.array(BLINDEXT[key]) - np.array([r["tvdec"] for r in EXT[key]])
        rows.append((name, emean(EXT[key], "tvdec"),
                     float(np.mean(BLINDEXT[key])), float(d.mean()), d))
for name, a, b, dm, d in rows:
    rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out.append(f"{name} & {a:.2f} & {b:.2f} & {dm:+.2f} [{lo:+.2f}, {hi:+.2f}] \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "cross.tex").write_text("\n".join(out) + "\n")

# ---- ablation table (specialists/blind/soft) ----
def mm(model, dose, key="psnr"):
    if model == "scr":
        return f"{np.mean(SCR2[str(dose)]):.2f}"
    v = MATCH.get(model, {}).get(str(dose), {}).get(key)
    return "--" if v is None else f"{v:.2f}"
out = [r"\begin{tabular}{lccccc}", r"\toprule",
       r"Dose & Specialist (under-budget) & Mixed-dose (base) & Conditioned (base) & Scrambled & TV dec \\",
       r"\midrule"]
for dose in [3, 30, 100]:
    out.append(f"$P={dose}$ & {mm('specialist', dose)} & "
               f"{mm('blind', dose)} & {mm('dcond', dose)} & "
               f"{mm('scr', dose)} & {mean(dose, 'tv_deconv'):.2f} \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
if CS:
    out += [r"\par\vspace{8pt}",
            r"\begin{tabular}{lccccc}", r"\toprule",
            r" & Specialist (strong) & Mixed-dose (strong) & Conditioned (strong) & Scrambled (strong) & TV dec \\",
            r"\midrule"]
    for dose in [3, 30, 100]:
        c = float(np.mean([r["psnr"] for r in CS[str(dose)]]))
        sp = (SPECSTRONG[str(dose)]["psnr"] if SPECSTRONG else float("nan"))
        sc = (float(np.mean(SCRSTRONG[str(dose)])) if SCRSTRONG
              else float(np.mean(SCR2[str(dose)])))
        out.append(f"$P={dose}$ & {sp:.2f} & {bmean(dose, 'psnr'):.2f} & "
                   f"{c:.2f} & {sc:.2f} & {mean(dose, 'tv_deconv'):.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
out += [r"\par\vspace{8pt}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"$\sigma$ (pixels) & U-Net & TV deconv \\", r"\midrule"]
for sigma in ["1.5", "2.0", "2.5"]:
    out.append(f"{sigma} & {np.mean(BLINDSIG[sigma]):.2f} "
               f"& {np.mean(SIGD[sigma]):.2f} \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "ablation.tex").write_text("\n".join(out) + "\n")

print("wrote fair numbers and tables")
