# Analysis protocol (recorded before the final fair-baseline runs)

This document records the analysis protocol used for the final manuscript. It
is a retrospective record, not a formally filed prospective preregistration.
The exploratory adaptive-allocation analysis was added after the main
reconstruction results and is reported as a negative/limitation result in the
Supplementary Information, not as a main claim.

## Data and splits

- Main dataset: BBBC014 channel 1, 96 FOVs, split 56 train / 20 validation /
  20 test at the FOV level (seed 0). No FOV is shared.
- Validation FOVs: early stopping and all classical-baseline tuning.
- Test FOVs: never used for training, tuning, early stopping, target selection,
  or figure selection.
- External datasets: BBBC015, BBBC016, BBBC018; 20 FOVs each, zero-shot only.
- Segmentation validation: BBBC039 (official 100/50/50 split), foreground masks;
  Otsu applied identically to all methods.

## Forward model

`y = Poisson(P * (g_sigma * x)) / P`, sigma = 2 pixels, P in
{3, 10, 30, 100, 300} photons per pixel at 8-bit full scale. Same PSF for all
methods.

## Primary endpoint

Protocol dose: the smallest P in the grid at which at least 90% of validation
FOVs reach a target PSNR, SSIM, or foreground PSNR; the selected dose is then
applied unchanged to held-out test FOVs. Primary targets: PSNR 30/32/34,
SSIM 0.85/0.90/0.92, FG 22/24/26. The primary baseline is TV-regularized
deconvolution with the known PSF (ADMM, 30 iterations, lambda tuned on
validation at each dose). TV denoising, NLM, Wiener, and Richardson-Lucy are
secondary baselines.

Correction note (2026-11-08): an earlier version of this paragraph said "90% of
test FOVs". The protocol selects the smallest dose on validation FOVs and then
applies it unchanged to test FOVs; all reported protocol factors use this
validation-selected definition. The test-oracle dose is a separate upper bound.


## Metrics

PSNR, SSIM, foreground PSNR (top 10% of GT), and nuclei Dice/IoU on BBBC039.
Main comparison: ten spread 128x128 patches per FOV, aggregated per FOV; 20 test
FOVs. Paired Wilcoxon and FOV-cluster bootstrap 95% CI. A result is reported only when the FOV-level test
and CI agree.

## Ablations

- Main model: mixed-dose U-Net trained on {3,10,30,100,300}, no dose input,
  all-dose validation.
- Dose-conditioned variant and per-dose specialists (3, 30, 100 photons).
- Phase-scrambled training control.
- Dose-only model at sigma 1.5, 2.0, 2.5 (PSF robustness).

## What was exploratory

- The BBBC039 downstream segmentation check was added after the restoration
  benchmark; with per-dose tuned TV its gain is only ~0.02 Dice, so it is
  reported as a secondary/limitation check, not a main endpoint.
- The adaptive per-FOV allocation analysis (preview + uncertainty + ridge/kNN)
  was exploratory. It did not consistently beat the best fixed dose and is
  reported in the SI as a negative result.
- The dose+PSF-conditioned model was exploratory; it did not improve over the
  dose-only model and is not part of the main claim.
- The BM3D subset test was exploratory and unstable; it is not used.

## Stop rule

If a fair deconvolution baseline (TV with the known PSF) had matched the
network at all targets, the reconstruction claim would have been withdrawn. It
did not: the strong mixed-dose network improved by +0.93 to +1.45 dB over fair
TV deconvolution, and with validation-selected doses saved 3-3.3x photons at
four targets while tying at three. Dose conditioning was tested as an ablation
and is optional (within 0.05 dB); no conditioning-necessity claim is made.

## Real-data check (added after the simulation study)

This check was added after the simulation study and the Round 3 review; it was
not part of the original preregistration. Its analysis was fixed before the
held-out comparison was run.

- Data: AI4Life-MDC25-Nuclei (Zenodo 15624741, CC BY 4.0). The 2,437 small
  pairs (after excluding 20 IDs shared numerically with the large-field list)
  are split by image ID into 1,951/243/243 train/validation/test.
- Final endpoint: PSNR and SSIM on all 243 held-out test pairs against BM3D,
  NLM, TV, and the noisy input (the initial 40-pair run was a compute-budget
  choice; see revision note). Classical parameters are tuned on five validation
  pairs over extended grids; test pairs are not used for tuning or early
  stopping.
- Statistics: paired Wilcoxon and win count over the 243 pairs.
- Cross-source check: the 20 large BBBC006 fields in the same release are
  evaluated with the same weights and parameters. This is a negative control
  for source transfer, not a headline endpoint.
- Scope: the real data test restoration within a matched acquisition protocol;
  they do not validate the simulated dose sweep or a cross-source claim.

## FMD second-source check (added 2026-11-07)

This is a retrospective secondary check, not part of the original
preregistration. The FMD subset is extracted at one frame per field, split by
field and stratified by microscope/channel/level into 1,396/301/301. The
comparison uses a pre-specified 150 of the 301 held-out fields (four per
stratum plus six fifth fields, selected before prediction; see the 2026-11-08
note). Classical parameters are tuned per microscope on five validation
fields; the FMD U-Net is retrained. The remaining 151 fields were not evaluated
with BM3D because of its cost, so this is a secondary check and not a headline
endpoint.

## Revision note (2026-11-07): fairness fixes and final analysis set

After the first real-data analysis, two problems were found and corrected
before the final numbers were reported:

- The initial real comparison had no affine calibration for BM3D/NLM/TV, while
  the supervised network was trained against an independently normalized clean
  image. A single global gain/offset per classical method is now fitted on the
  five validation pairs and applied to the held-out predictions. This is the
  fair comparison reported in the main text; raw values remain in the SI.
- The main real endpoint was changed from the first 40 held-out pairs to all
  243 held-out pairs. The 40-pair subset was a compute-budget choice, not a
  performance filter; the full set is used in the final analysis and no pair
  was excluded based on its result.

The earlier 36-field FMD check was labeled preliminary in the SI because its
classical parameters were tuned on a validation set dominated by one microscope
stratum. It has been superseded by the pre-specified 150-field subset with
per-microscope tuning described below and is not used as a headline endpoint.
The simulation baseline was also retuned (joint weight/rho/iteration search
with reflect padding) before the final strong-model comparison; all test
results use that fixed fair baseline.

## FMD 150-field subset (pre-specified 2026-11-08)

The earlier FMD check used one held-out field per microscope/channel/level
stratum (36 fields). To strengthen it without evaluating all 301 test fields,
we pre-specified a 150-field subset before any new baseline or network
prediction was computed. Within each of the 36 strata (microscope x channel x
noise level), we take the first four test fields in lexicographic field-ID
order; the fifth field from the first six strata in lexicographic order fills
the subset to 150. The selection rule is performance-independent and is stored
in `prereg/fmd_subset150.json`. Baselines remain tuned on the original
validation split; no test field is used for tuning or model selection.
