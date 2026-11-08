# Dose-agnostic neural deconvolution for low-dose fluorescence microscopy

A study of dose-agnostic neural deconvolution on public microscopy images,
with a real noisy/clean validation and a small downstream segmentation check.
The task is to recover a high-SNR fluorescence field of view from a
Poisson-corrupted, Gaussian-blurred image at P = 3–300 full-scale units
(median 0.3–34 detected photons per pixel).

## Main result

A single mixed-dose U-Net, evaluated without a dose input, improves over the
best tuned classical method (TV-ADMM deconvolution with jointly tuned weight
and ADMM parameters) by **+0.93, +1.36, and +1.37 dB** PSNR at P=3, 30, and 300
full-scale units on 20 held-out BBBC014 fields of view (BBBC014 channel 1;
20/20 FOVs improve at every dose; gain range 0.93–1.45 dB across P=3–300). With
doses selected on validation fields, it uses **3–3.3× fewer photons** at 90%
FOV success for PSNR 32 and 34 dB, SSIM 0.90, and foreground PSNR 26 dB, with
ties at SSIM 0.85 and 0.92 and foreground PSNR 24 dB.

A budget-matched dose-conditioned variant is within 0.05 dB of the strong
mixed-dose model but requires the photon count at inference; conditioning is
therefore optional. A phase-scrambled control falls below TV deconvolution,
and ensemble uncertainty detects failing fields with AUC 0.96 at 3 and 10
photons.

On **243 held-out real noisy/clean pairs** from the AI4Life-MDC25 release
(Jacquemet source, Zenodo 15624741, CC BY 4.0), the same U-Net design (48 channels) trained
on real pairs reaches **29.01 dB PSNR / 0.877 SSIM** versus affine-calibrated tuned BM3D
24.83 dB / 0.559, NLM 24.78 / 0.551, and TV 24.51 / 0.526 (+4.18 dB over BM3D,
95% CI 3.94–4.42, 240/243 wins, Wilcoxon p = 2.2e-41). This is an in-source result: the model fails
to transfer to the 20 large BBBC006 fields in the same release, which differ in
bit depth, saturation, and noise model.

A second real source (FMD; a pre-specified 150-field subset of 301 held-out
fields, spanning three microscopes, three channels, and four noise levels) is
reported in SI S7b. With per-microscope-tuned baselines the U-Net reaches
31.74 dB / 0.845 SSIM versus BM3D 28.07 / 0.704 (+3.67 dB, 147/150); the gain
is positive in all 36 strata. Because the FMD classical outputs are not
affine-calibrated, this is a secondary upper bound, not a headline.

A downstream BBBC039 nuclei-mask check with per-dose tuned TV found only a
small Dice gain (**0.795 vs 0.773 at P=1**; 0.844 vs 0.824 at P=3). The large
gain against an untuned TV weight was an under-regularization artifact. The
segmentation result is therefore reported as a secondary/limitation check, not
a headline.

The dose sweep is simulated: public high-SNR images are degraded with a known
Gaussian PSF and Poisson noise. Real data support restoration within a matched
acquisition protocol, not the cross-source dose claim. See `paper/main.pdf`
and `paper/si.pdf`.

## Layout

```
src/                 forward model, data, baselines, training, evaluation
results/             JSON result files (no fabricated numbers)
checkpoints/         trained models
figures/             figure PDFs/PNGs
paper/               main.tex, si.tex, generated numbers/tables, PDFs
datasets/            downloaded public data (not committed in full)
```

## Quick start

```bash
python -m pip install -r requirements.txt
# download the four BBBC archives (see DATASETS.md)
python src/train_unet.py --dataset bbbc014 --patches 24000 --epochs 100 \
  --val-patches 4 --mixed-doses --val-all-doses \
  --out checkpoints/bbbc014_blind3.pt
python src/eval_blind_controls.py
python src/bench_tvdeconv.py
python src/make_fair_assets.py
python src/make_fig1.py && python src/make_fig2.py && \
python src/make_fig3.py && python src/make_fig4.py

# real noisy/clean validation (AI4Life-MDC25; see DATASETS.md)
python src/train_real.py --patches 8000 --epochs 40
python src/tune_real_classical.py --n-val 5 --workers 8
python src/eval_real_small_par.py --n 0 --workers 12
python src/eval_real_large_par.py --n-patch 8 --workers 12
python src/make_real_assets.py && python src/make_fig_real.py

cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

See `REPRODUCE.md` for the full command set, `DATASETS.md` for licenses and
checksums, `CLAIM_MAP.md` for claim-to-evidence links, and `REVIEW_LOG.md` for
the review history.
