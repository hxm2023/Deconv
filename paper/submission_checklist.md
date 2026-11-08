# Submission checklist

## Manuscript
- [x] `paper/main.pdf` (12 pages), `paper/si.pdf` (10 pages), and
      `paper/main_elsevier.pdf` (22 pages, line numbers) compile.
- [x] Title/abstract state the positive result, the real-data check, and the
      simulation scope.
- [x] All result numbers generated from `results/*.json` via `paper/numbers.tex`,
      `paper/real_numbers.tex`, and `paper/tables/*`.
- [x] `CLAIM_MAP.md` links each claim to a JSON/table/figure.
- [x] `REVIEW_LOG.md` records the fair-baseline and untuned-TV corrections.
- [ ] Author names, affiliations, corresponding author, funding, conflicts, CRediT
      (template: `paper/AUTHOR_INFO.md`; must be filled by the authors).
- [x] Elsevier `elsarticle` version synced from `main.tex` with line numbers,
      keywords, declarations, and `elsarticle-num` references.
- [ ] Fill author/affiliation placeholders and the four declaration blocks in
      `main_elsevier.tex` before upload.
- [x] Cover letter draft (`paper/cover_letter.md`); signature and reviewer
      suggestions to be added by the corresponding author.

## Data and code
- [x] Public data only: BBBC014/015/016/018/039 and AI4Life-MDC25.
- [x] Licenses and SHA-256 checksums in `DATASETS.md` and SI S1/S5.
- [x] FOV-level split; no test FOV used for training, tuning, early stopping, or
      figure selection.
- [x] Code, checkpoints, JSON results, figure scripts, and `REPRODUCE.md`.
- [ ] Archive/release repository (remove any local-only data before upload).

## Claims
- [x] Main claim: mixed-dose U-Net vs fair TV-ADMM; +0.93/+1.36/+1.37 dB at
      P=3/30/300 (0.93-1.45 dB across the sweep); validation-selected 3-3.3x
      savings at four targets, three ties, two non-transferring selections;
      cross-dataset positive but smaller after per-dataset TV tuning.
- [x] Real-data claims: +4.18 dB over affine-calibrated BM3D on 243 nuclei
      pairs (240/243 wins), stated as in-source restoration; strong-model
      cross-source large-field failure in SI; FMD is a secondary 150-field SI check.
- [x] Conditioning is optional: budget-matched conditioned model within 0.05 dB,
      budget-matched specialists within 0.1 dB; blind model is the main method.
- [x] Patch-draw sensitivity (5 draws, 4/10 patches) and affine-refit stability
      (5/20/50/243 validation pairs) are reported in SI.
- [x] Physical Airy-PSF check (Gaussian-trained vs PSF-mixed) and
      budget-matched DnCNN are reported; FMD is a secondary 150-field SI check.
- [x] Downstream BBBC039 segmentation is a small/limitation result, not a headline.
- [x] Adaptive allocation negative result is in SI.
- [x] No hardware, controlled real low-dose, or clinical claim.

## Venue
- Primary recommendation: **Ultramicroscopy** (Q1/Q2, specialized microscopy).
- Alternative: Optics Express (strict reviewer rated the current version borderline;
  needs real low-dose/real-noise validation to strengthen).
- Fallback: Journal of Microscopy / Measurement Science and Technology (Q3).
