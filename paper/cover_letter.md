# Cover letter (draft)

[Date]

Dear Editor,

We submit the manuscript "Dose-agnostic neural deconvolution for low-dose
fluorescence microscopy" for consideration as a Research Paper in
*Ultramicroscopy*.

Low photon budgets protect live samples but degrade the images from which
structure must be recovered. Classical deconvolution must be re-tuned for each
noise level. We show that a single U-Net trained on mixed doses, with no
photon-count input at inference, reconstructs fluorescence images across a
100-fold photon range. On 20 held-out BBBC014 fields of view it improves over
tuned TV-ADMM deconvolution by 0.93, 1.36, and 1.37 dB at P = 3, 30, and 300
full-scale units. With doses selected on validation fields, it uses 3-3.3x
fewer photons at a matched 90% field-of-view success rate at PSNR 32 and
34 dB, SSIM 0.90, and foreground PSNR 26 dB, with ties at SSIM 0.85 and 0.92
and foreground PSNR 24 dB. A
budget-matched dose-conditioned variant is within 0.05 dB of the blind model
but needs the photon count, so conditioning is optional. With an in-focus Airy
PSF, a PSF-mixed variant retains gains of 1.5, 1.4, and 1.3 dB at P = 3, 30,
and 300, where the Gaussian-trained model loses its high-dose advantage. The model transfers
to three external public datasets with smaller but positive gains and provides
ensemble uncertainty that flags unreliable fields (AUC 0.96).

We also report a real-data check on the public AI4Life-MDC25 release. On 243
held-out real noisy/clean pairs, the same U-Net design (48 channels) trained on real pairs
reaches 29.01 dB PSNR and 0.877 SSIM versus 24.83 dB and 0.559 for
affine-calibrated tuned BM3D (+4.18 dB, 240/243 wins, Wilcoxon p = 2.2e-41).
The same weights fail on the 20 large BBBC006 fields in the same release
(15.83 versus 17.05 dB for BM3D), which we report as a cross-source limitation.
A second real source (FMD) was evaluated on a pre-specified 150-field subset
with per-microscope-tuned baselines; it gives +3.67 dB over BM3D but is not
affine-calibrated, so it is a secondary SI upper bound. The dose sweep itself
is simulated with a known Gaussian PSF and Poisson noise; we do not claim a
hardware, controlled real low-dose, or clinical result.

The work fits *Ultramicroscopy* as a quantitative microscopy study with a clear
positive result, a fair classical baseline, an explicit real-data check, and
honest scope statements. All data are public (BBBC014/015/016/018/039 and
AI4Life-MDC25), with licenses and checksums in the Supplementary Information.
Code, checkpoints, and JSON result files are available.

This manuscript is not under consideration elsewhere. All authors have approved
the submission and declare [no competing interests / competing interests as
stated in the manuscript].

Sincerely,

[Corresponding author name]
[Affiliation]
[Email]
[Signature]
