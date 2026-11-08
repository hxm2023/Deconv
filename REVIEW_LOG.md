# Review log

## FINAL STATUS (fair/strong manuscript)

The current manuscript reports a dose-agnostic mixed-dose U-Net with a
controlled simulation and an in-source real noisy/clean validation. The main
model is trained on {3,10,30,100,300} full-scale Poisson units, receives no
dose input, and is evaluated without metadata. Against fair, jointly tuned
TV-ADMM deconvolution it gains +0.93/+1.12/+1.36/+1.45/+1.37 dB at
P=3/10/30/100/300 (20/20 FOVs). With validation-selected doses it saves
3-3.3x photons at PSNR 32/34 dB, SSIM 0.90, and foreground PSNR 26 dB; three
targets tie, and two validation selections do not transfer. A budget-matched
conditioned model is within 0.05 dB and budget-matched specialists within
0.1 dB, so conditioning is optional. A budget-matched blind phase-scrambled
control stays below TV. On 243 held-out real AI4Life-MDC25 pairs the 48-channel
model reaches 29.01 dB / 0.877 SSIM versus 24.83 / 0.559 for affine-calibrated
BM3D (+4.18 dB, 240/243); raw values and calibration stability are in the SI.
The same weights fail on the 20 large BBBC006 fields (15.83 vs 17.05 dB), a
cross-source domain shift. The dose sweep remains simulated. An earlier
BBBC039 segmentation check with per-dose tuned TV found only a small Dice gain
and is a secondary/limitation result. FMD is a preliminary SI check. A
dose/task identifiability probe (dose 97.9% linear / 100% CNN; task 79.5% /
75.8%) supports the interpretation that the blind model infers the dose
internally.

## Round 2 — second strict review

The reviewer found that the first "blind" model had been trained only at 300
photons because `build_training_set` sampled mixed doses only when the dose
channel was enabled. It also found: TV denoising is marginally better than TV
deconvolution at 300 photons; the adaptive JSONs were not archived; the SI
extended-baseline values came from a different patch set; PSF and external TV
parameters lacked provenance; speed numbers were hard-coded; cross-dataset
CIs/n were missing; and the figures/tables had label issues. All are addressed
or explicitly disclosed in the final text/SI.

## Round 3 — final review (current manuscript)

Recommendation: **major revision / borderline reject for Optics Express**, with
J. Microscopy or Ultramicroscopy suggested if the work is not reframed/expanded
with real-noise validation. The reviewer accepted the fair-baseline rebuild and
the current mixed-dose model, but flagged (1) figure selection on test FOVs,
(2) stale evidence-chain docs, (3) photon units, (4) external channel/sampling,
(5) coarse protocol grid, and (6) the incremental nature of a blind mixed-dose
U-Net. A downstream BBBC039 check was added but, after per-dose TV tuning, the
segmentation gain is only ~0.02 Dice; it is reported as a secondary/limitation
result, not a headline. The final version uses validation FOVs for the example figures, corrects
BBBC015 channel 0 and BBBC018 green-channel loading, adds photon-unit and
read-noise sections, and updates the claim map. Real paired low-dose data are
not available in this study; the manuscript is framed as a controlled
simulation benchmark. (Superseded by Round 4: an in-source real noisy/clean
check was added; the dose sweep remains simulated.)

## Round 1 — strict in-house reviewer (pre-fair-baseline manuscript)

Recommendation: **major revision, borderline reject.**

The reviewer's central objection was that `tv()` and `nlm()` in
`src/baselines.py` are denoisers, not deconvolution: they never apply the PSF.
The manuscript therefore compared a deblurring network against pure denoisers.
Additional findings: parameter grids saturated at their boundaries; the paper
called patch-level measurements "full FOV"; there was no noisy-input baseline;
the blind-model and specialist controls were not compute-matched; the PSF-mismatch
claim was untested; the uncertainty AUC at 30 photons was based on a single
failure; the external datasets were described imprecisely; the adaptive result
referenced a non-existent SI; and the figures were too small at print width.

## Response

1. **Fair deconvolution baseline.** Implemented TV-regularized deconvolution
   with ADMM and the known PSF (`src/baselines.py: tv_deconv`). Retuned lambda
   on validation FOVs at each dose (0.2, 0.08, 0.04, 0.04, 0.04 for
   3/10/30/100/300 photons). All headline numbers now use this baseline.
   Result: the U-Net advantage is +1.02, +1.83, and +3.42 dB at 3, 30, and 300
   photons (previously +1.9, +2.4, +3.3 against TV denoising). The 3-10x
   protocol savings remain at PSNR 32 dB, SSIM 0.90-0.92, and foreground PSNR
   22-26 dB; at PSNR 30 dB and SSIM 0.85 the fair baseline ties.
2. **Boundary grids.** Extended TV, NLM, Wiener, and RL grids. NLM saturates at
   28.6/33.4 dB (3/30 photons); Wiener reaches 29.2/33.5 dB and RL 29.3/33.6 dB
   with the extended grids. All remain below TV deconvolution and the network.
3. **Patch-level wording.** The manuscript now says ten spread 128x128 patches
   per FOV, aggregated per FOV; no "full FOV" claim.
4. **Noisy baseline.** Added to Table 1 (14.60, 24.10, 32.48 dB at 3, 30, 300).
5. **Blind model (superseded).** The first retrain still sampled only 300
   photons because mixed-dose sampling was tied to the dose-condition flag.
   After adding `--mixed-doses`, the true mixed-dose model reaches
   32.05/36.27/38.07 dB at 3/30/100 versus 32.20/36.44/38.29 for the
   conditioned variant. The conditioning-necessity claim is withdrawn; the
   mixed-dose model is now the main method.
6. **Specialists.** Retrained the 30-photon specialist with the same 16k/60
   budget as the 3- and 100-photon specialists. The conditioned model matches
   or beats all three (32.20 vs 32.18; 36.44 vs 36.43; 38.29 vs 37.95).
7. **PSF mismatch.** Tested the dose-only model at sigma 1.5/2.0/2.5: it remains
   above TV deconvolution (36.77/36.43/35.49 vs 35.28/34.61/33.95). An explicit
   PSF-conditioning channel did not improve accuracy; the PSF-conditioning claim
   was removed.
8. **Uncertainty.** AUC reported only at 3 and 10 photons (0.96/0.96, n=15/11).
   At 30 photons only one FOV fails; no AUC is reported.
9. **External data.** Corrected descriptions: BBBC015/016 are U2OS
   cytoplasm-to-nucleus assays; BBBC018 is a different cell type and actin
   stain. External gains over TV deconvolution are +0.47 (BBBC015), +1.46
   (BBBC016), and +0.16 dB (BBBC018), all with CIs excluding zero but smaller
   than in-domain.
10. **Adaptive result.** Added `paper/si.pdf` with the adaptive table and the
    target-dependent, non-robust outcome; the main text marks it as SI.
11. **Figures.** Rescaled all figures to the 6.5-inch text width with readable
    fonts; added the noisy baseline, fair TV-deconv curves, blind model, and PSF
    robustness panel.
12. **BM3D.** Removed from the manuscript; the public implementation was
    unstable at very low counts in our hands. NLM and TV deconvolution provide
    the non-local and variational comparisons.

## Remaining known limitations (reported in the paper)

- Simulated low-dose sweep; real AI4Life pairs support in-source restoration
  only, and the large BBBC006 source is a cross-source failure.
- Gaussian and in-focus Airy PSFs tested; arbitrary aberrations and depth-varying PSFs are not.
- Single training dataset; external gains are smaller.
- Learned restoration can hallucinate; uncertainty is a quality-control signal,
  not a guarantee.
- The conditioned model uses 24k patches versus 16k for the specialists, but
  replaces three models; total training cost is lower.

## Round 2

Pending after the fair-baseline rewrite. The manuscript and SI are in
`paper/main.pdf` and `paper/si.pdf`.

## Round 4 — real noisy/clean validation

Added after the Round 3 review, which asked for real-noise evidence.

1. **First real run (negative).** A U-Net trained on the AI4Life-MDC25 small
   pairs (1,951 training images) reached 14.68 dB on the 20 large 1024×1024
   fields, below tuned BM3D (17.01 dB) and NLM (16.82 dB).
2. **Diagnosis.** The small and large subsets are different acquisition sources,
   not crops of one another: same-ID small/large images correlate at 0.16, the
   small references reach 4095 counts, and the large references are 16-bit and
   clipped at 65535. The release combines 2,457 Jacquemet noisy-nuclei pairs
   with 20 BBBC006 fields. Multi-scale correlation (0.805 at sigma=2 for field
   1000) confirmed that the large pairs are valid but strongly saturated.
3. **Leakage checks.** No near-duplicate leakage: the maximum thumbnail
   correlation is 0.26 for validation and 0.43 for test against 400 training
   images; consecutive IDs are uncorrelated.
4. **In-source comparison.** On 40 held-out small pairs, with classical
   parameters tuned on validation pairs over extended grids (BM3D sigma=0.3,
   NLM h=0.2, TV w=0.24; all interior optima), the U-Net reaches 28.00 dB /
   0.839 SSIM versus BM3D 20.92 / 0.348, NLM 20.93 / 0.344, TV 20.25 / 0.322,
   and noisy 16.10 / 0.104. Paired gain over BM3D: +7.07 dB, 39/40 wins, exact
   Wilcoxon p=3.6e-12; over TV: +7.75 dB. The full 243-pair evaluation superseded
   this subset: 28.70/0.872 vs BM3D 21.30/0.359, +7.40 dB, 241/243, p=1.4e-41.
   The 40-pair references have ground-truth
   standard deviation >=0.078, so the result is not driven by empty fields.
5. **Cross-source check.** With the same tuned parameters, the large BBBC006
   fields give U-Net 14.68 dB versus BM3D 17.05, NLM 16.89, and TV 16.45. The
   failure is reported as a domain shift and bounds the claim to matched
   acquisition.
6. **Decision.** The real-data claim is stated as in-source restoration, not as
   real low-dose validation; the dose sweep remains simulated. The paper title
   was changed from "a controlled simulation benchmark" to a method-first title,
   and the target journal is Ultramicroscopy.

Evidence: `results/real_small_test40.json`, `results/real_small_tuning.json`,
`results/real_large_fair.json`; main Table 3 and Fig. 4; SI S7.

## Round 5 — strict Codex review and revision

A fresh strict Q2 referee review of the 9-page draft returned **Major revision**.
Required items and responses:

1. **All 243 real test pairs.** The full evaluation supersedes the 40-pair
   subset: U-Net 28.70 dB / 0.872 SSIM vs BM3D 21.30 / 0.359, NLM 21.28 /
   0.354, TV 20.58 / 0.333; +7.40 dB over BM3D, 241/243 wins, p=1.4e-41. The
   two losses are the two weakest-signal pairs (GT std 0.047 and 0.078).
2. **Second supervised deep baseline.** A 17-layer DnCNN trained on 4,000
   patches/20 epochs reaches 27.57 dB / 0.807 SSIM: 6.27 dB above BM3D but
   1.13 dB below the U-Net, so the gain is not specific to one architecture.
3. **Full leakage audit.** All 243 test vs all 1,951 training images: maximum
   thumbnail correlation 0.55, no pair above 0.8, top full-resolution
   correlation 0.06 (`results/real_leakage_full.json`).
4. **Protocol and AUC uncertainty.** FOV-bootstrap factor CIs and AUC CIs are
   in SI S2b/S4 and Fig. 3. The abstract now qualifies the 3-3.3x/10x factors.
5. **Writing and figures.** Abstract shortened; Methods truncation fixed;
   photon units defined in the Introduction; limitations split into labeled
   paragraphs; Fig. 1 adds a test FOV, scale bar, zoom inset, and colorbar;
   Fig. 2(d) replaces n/a bars with markers and prints factor CIs; Fig. 3(d)
   is a re-acquisition curve; Fig. 4 adds a cross-source failure row and
   scale bars.

A second fresh Codex review returned **Minor revision**: eight of nine
required fixes were verified; the remaining Fig. 3 and Fig. 2(d) items were
then corrected (n=20, scale bar, uncertainty colorbar; n/a markers and factor
CIs). The reviewer's remaining non-blocking nit (Fig. 1 colorbar tick) was
also fixed. The manuscript is now at the "acceptable after minor edits" stage
for Ultramicroscopy.

**Second real source (FMD).** To address the single-real-source limitation, the
AI4Life-MDC25-FMD release (Zenodo 15761900, CC BY 4.0) was downloaded and
verified (MD5 db2259fd8ae551ebc4f4b22336681369; SHA-256
5110e958bef215e42649c2477855ea11f402842e11523ba0aebf05a634d7eb8f). One frame
from each of 1,998 fields was extracted and split by field, stratified by
microscope/channel/level, into 1,396/301/301. On 36 stratified held-out fields
(one per microscope/channel/level), the U-Net reaches 31.90 dB / 0.849 SSIM
versus tuned BM3D 27.88 / 0.724, NLM 27.04 / 0.702, and TV 27.26 / 0.716:
+4.02 dB over BM3D, 35/36 wins, Wilcoxon p=8.7e-11. The real-data advantage is
therefore not specific to the nuclei source.

**Final reviewer outcome.** After the two blocking fixes (Fig. 1 horizontal
error colorbar with all ticks visible; `results/real_selfsup_all.json` restored)
and the non-blocking fixes (Fig. 2 caption, nuclei 95% CI, 0.19-1.4 dB range,
denoising-only clause, zero-shot sentence relocation), a fresh Codex check
returned **PASS — both blocking items fixed; no new blockers.** Combined with
the preceding verdict ("Minor revision — scientifically acceptable for a Q2
fast-review journal; with the two blocking items fixed, acceptable for
publication"), the manuscript has cleared the agent review for a Q2
physics/microscopy venue. Final build: main.pdf 12 pages, si.pdf 7 pages,
main_elsevier.pdf 19 pages.

## 2026-11-07: fairness audit and strong-model revision

A code/data audit found three sources of inflated margins, and the manuscript
was rebuilt around fair replacements:

1. **TV baseline under-tuned.** The original TV-ADMM used a fixed `rho=1`,
   30 iterations, and circular padding. The fair baseline jointly tunes the
   weight, `rho` in {0.3,1,3}, and iterations in {30,60,100} on validation and
   uses reflect padding. This raised TV deconvolution to 31.34/33.62/35.34/
   37.16/38.63 dB at P=3/10/30/100/300. All pairwise gains were recomputed
   against this baseline.
2. **Independent normalization advantage on real pairs.** The U-Net predicts an
   independently normalized clean image, while BM3D/NLM/TV do not. One global
   affine gain/offset per classical method is now fitted on five validation
   pairs and applied to held-out test predictions. This reduced the nuclei gain
   from +7.40 to +3.88 dB in the base-budget model and to +4.18 dB after
   retraining the real model at 48 channels/16k patches/80 epochs
   (29.01 dB / 0.877 SSIM vs affine BM3D 24.83 / 0.559; 95% CI 3.94–4.42,
   240/243 wins, p=2.2e-41). The raw gain is reported separately in the SI.
3. **FMD baseline tuned on a single stratum.** The FMD baselines were tuned on
   five validation pairs that all came from one microscope, so the reported
   +4.02 dB mixed per-microscope effects. FMD is now demoted to a preliminary
   SI check with an explicit caveat; it is not in the abstract, main table, or
   headline claims.

The strong simulation model (64 channels, 48k patches, 150 epochs) was trained
and evaluated against the fair TV baseline: 32.27/34.74/36.70/38.61/40.01 dB,
gains of +0.93/+1.12/+1.36/+1.45/+1.37 dB. A budget-matched strong
dose-conditioned model reaches 32.29/36.71/38.66 dB at P=3/30/100, i.e. within
0.01–0.05 dB of the blind model, confirming that photon-count conditioning is
optional. The base-budget ablation (specialists/conditioned/scrambled) is kept
in the SI with a caption that states its budget, and the budget-matched strong
comparison is shown separately.

Other corrections: the PSF table now reports the mean over FOVs rather than the
first FOV; read-noise was rerun with the strong model and fair TV; the extended
baseline table now lists the fair TV values; and all downstream README,
highlight, cover-letter, claim-map, reproduction, and SI text was updated to the
new numbers. The dose sweep remains simulated; the real-data claim remains
in-source and does not transfer zero-shot to BBBC006. A budget-matched blind
phase-scrambled control (64 channels, 48k patches, 150 epochs) reaches
30.19/34.46/36.18 dB at P=3/30/100, below both fair TV
(31.34/35.34/37.16) and the real model (32.27/36.70/38.61), so the gain is not
explained by capacity alone.

**Final strict review (2026-11-07).** A second-round review of the fair/strong
manuscript returned **minor revision** with six must-fix items: Methods still
defined the protocol dose on test FOVs; the protocol table printed factors for
rows whose selected dose missed the test criterion; Fig. 2(d) attributed the
FG22 failure to the U-Net instead of TV; the test-oracle range was written as
3.3-fold instead of 3-3.3-fold; CARE/FFDNet were described as both
noise-conditioned; and the PSF ablation was misdescribed as a 300-photon
model instead of a dose+PSF-conditioned model. All six were fixed, and a final
verification returned **ACCEPT with no remaining must-fix blockers**. Final
build: main.pdf 11 pages, si.pdf 8 pages.

## 2026-11-08: writing, figures, and journal-format pass

Codex was used as an external writing/figure referee over three rounds. The
abstract and introduction were rewritten as narrative prose (no contribution
list or rhetorical questions), the photon-unit definition was corrected to a
full-scale Poisson scale with mean detected photons stated separately, and the
Conclusions were tightened. Budget-matched specialists were trained
(32.36/36.72/38.68 dB vs blind 32.27/36.70/38.61, within 0.1 dB), a five-draw
patch-draw check was added (gains 0.80-1.01, 1.30-1.45, 1.29-1.43 dB), and the
affine calibration was refitted on 5/20/50/243 validation pairs (coefficient
change <7%; 20-pair gain 3.74-3.98 dB). Four main figures were redrawn:
Fig. 1 is a six-column qualitative grid with an aligned error colorbar, Fig. 2
adds a paired-gain panel and a validation-selected protocol dot plot, Fig. 3
has panel letters, a fit line, and a calibrated uncertainty map, and Fig. 4
uses percentile-selected pairs, panel letters, and an unclipped scatter. The
reference list was expanded to 32 verified works; the invalid Mohan et al.
PNAS DOI was rejected and the ICLR/arXiv record retained. The Elsevier version
is now generated from the same source with line numbers, `elsarticle-num`,
keywords, declarations, and tables forced in place. Codex's final verdict:
acceptable for submission to Ultramicroscopy, contingent only on filling the
author/affiliation and declaration placeholders. Final build: main.pdf 12
pages, si.pdf 9 pages, main_elsevier.pdf 22 pages.

## 2026-11-08: physical-PSF and budget-matched-baseline checks

Two additional checks were run after the writing/figure pass. First, the
Gaussian training PSF was replaced with an in-focus scalar Airy PSF (NA 1.4,
520 nm, 100 nm pixels; FWHM about 1.9 pixels), with TV retuned on validation
using the correct PSF. The Gaussian-trained model keeps a small advantage at
P=3/30 (+1.42/+0.41 dB) but loses 3.21 dB at P=300. A model trained on a 50/50
Gaussian/Airy mixture (same 64-channel/48k/150 schedule) reaches
32.71/37.61/41.55 dB versus TV 31.21/36.23/40.26 (gains +1.50/+1.39/+1.29,
20/20, CIs excluding zero) and retains 32.26/36.60/39.83 dB on the Gaussian
benchmark. Second, the DnCNN baseline was retrained with the same 16k
patches/80 epochs as the real U-Net; it reaches 28.80 dB/0.865 SSIM versus
29.01/0.877 for the U-Net (+0.21 dB for the U-Net, 95% CI 0.15-0.28,
210/243). This caps any architecture-specific claim while confirming that the
real-data gain over classical methods is shared by a second deep model. A
full 301-field FMD evaluation was attempted but stopped as impractical; FMD
remains a preliminary 36-field SI check. Codex reviewed the additions and
reported the numbers reproducible and the Q2 Ultramicroscopy verdict
unchanged. Final build: main.pdf 12 pages, si.pdf 9 pages, main_elsevier.pdf
22 pages.

## 2026-11-08: FMD 150-field pre-specified check

The 36-field FMD subset was expanded to 150 of the 301 held-out fields using a
rule fixed before prediction: within each of the 36 microscope/channel/level
strata, take the first four fields in lexicographic field-ID order, plus the
fifth field from the first six strata. The selection is stored in
`prereg/fmd_subset150.json`; no performance information entered it. Classical
baselines were tuned per microscope on five validation fields (BM3D sigma 0.12
CF, 0.15 TP, 8.0 WF, with the WF search extended to 20; NLM h=0.1; TV weights
0.08 CF and 0.12 TP/WF). On the 150 held-out fields the U-Net reaches 31.74
dB/0.845 SSIM versus BM3D 28.07/0.704, NLM 26.96/0.697, TV 27.01/0.707, and
noisy 25.62/0.584: +3.67 dB over BM3D (95% CI 3.27-4.12, 147/150 wins,
p=2.9e-26). The gain is positive in all 36 strata (group means 0.4-11.1 dB),
but FMD classical outputs are not affine-calibrated, so the result is a
secondary upper bound, not a headline. The 150-field evaluation took 1491 s on
8 workers; per-microscope BM3D tuning took longer because the WF optimum needs
a sigma of 8.0. Final build: main.pdf 12 pages, si.pdf 9 pages,
main_elsevier.pdf 22 pages.
