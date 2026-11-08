# Highlights (Ultramicroscopy, max 85 characters each)

- A single U-Net deconvolves across a 100-fold photon range without dose input.
- Beats tuned TV-ADMM by 0.93–1.45 dB; 3–3.3× fewer photons at matched quality.
- On 243 real held-out pairs it beats affine-calibrated BM3D by 4.18 dB in denoising.
- A PSF-mixed variant keeps the gain under a physical Airy PSF.
- The real gain is in-source and does not transfer across acquisition sources.

# Suggested cover-letter summary

The manuscript reports a dose-agnostic neural deconvolution method for low-dose
fluorescence microscopy. A single U-Net trained on mixed doses, with no
photon-count input at inference, improves over tuned TV-ADMM deconvolution by
0.93, 1.36, and 1.37 dB at P=3, 30, and 300 full-scale units on 20 held-out
BBBC014 fields of view (20/20 FOVs, paired CIs excluding zero). With doses
selected on validation fields, it uses 3–3.3x fewer photons at 90% FOV success
for PSNR 32 and 34 dB, SSIM 0.90, and foreground PSNR 26 dB, with ties at SSIM
0.85 and 0.92 and foreground PSNR 24 dB. A budget-matched dose-conditioned variant is within
0.05 dB of the blind model but needs the photon count, so conditioning is
optional. The model transfers to BBBC015/016/018 with smaller but positive
gains and provides calibrated ensemble uncertainty. On 243 held-out real
noisy/clean pairs from AI4Life-MDC25, the same U-Net design (48 channels) trained on real
pairs reaches 29.01 dB PSNR and 0.877 SSIM versus 24.83 dB and 0.559 for
affine-calibrated tuned BM3D (+4.18 dB, 240/243 wins). The same weights fail on
the 20 large BBBC006 fields in the same release (15.83 versus 17.05 dB for
BM3D), which we report as a cross-source limitation. A second real source
(FMD, pre-specified 150-field subset with per-microscope-tuned baselines) gives
+3.67 dB over BM3D but is not affine-calibrated, so it remains a secondary SI
upper bound, not a headline.

The dose sweep is simulation: public high-SNR images are degraded with a known
Gaussian PSF and Poisson noise. The real pairs validate restoration within a
matched acquisition protocol, not a controlled low-dose series or a
cross-source deployment. No hardware deformable-mirror or clinical claim is
made. All data are public (BBBC014/015/016/018/039 and AI4Life-MDC25), with
licenses and checksums in the SI. Code, checkpoints, JSON result files, and
figure/table generation scripts are available.

We believe the work fits Ultramicroscopy as a quantitative microscopy study
with a clear positive result, a fair classical baseline, and an explicit
real-data check. It is not under consideration elsewhere.
