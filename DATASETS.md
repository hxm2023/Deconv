# Datasets

All data are public. The BBBC low-dose measurements are synthesized from the
high-SNR references; AI4Life-MDC25 Nuclei and FMD supply native real
noisy/clean pairs. No proprietary or unpublished companion data are used.

| Dataset | URL | Role | FOVs used | Image | License | SHA-256 |
|---|---|---|---|---|---|---|
| BBBC014 channel 1 | https://data.broadinstitute.org/bbbc/BBBC014/BBBC014_v1_images.zip | train/val/test (56/20/20) | 96 total | 1360×1024, 8-bit | CC BY 3.0 | `223c3230c164c8295f4626f82e4bd9fe394d8e9339a4f82faf697fdd9b835341` |
| BBBC015 | https://data.broadinstitute.org/bbbc/BBBC015/BBBC015_v1_images.zip | external test (20) | 144 FOVs (channel 0) | 1000×768, 8-bit | CC BY 3.0 | `8c7c2abc579ff34173b49ccd06a167e1ad682f35506aa3f6e333d0336e1bad23` |
| BBBC016 GFP (d0) | https://data.broadinstitute.org/bbbc/BBBC016/BBBC016_v1_images.zip | external test (20) | 72 per channel | 512×512, 8-bit | CC BY 3.0 | `866ea3e6229a0f6fa2c9a18852b3274f7617326ecfae15d40f7f5e1123df7dd1` |
| BBBC018 actin (green channel) | https://data.broadinstitute.org/bbbc/BBBC018/BBBC018_v1_images.zip | external test (20) | 56 | 512×512, 8-bit | CC BY-NC-SA 3.0 | `e2996b78e16de2eff74db649c1fe5335edfd82ded061e7e6afeadf27caa09df1` |
| BBBC039 images | https://data.broadinstitute.org/bbbc/BBBC039/images.zip | segmentation (official 100/50/50) | 200 | 520×696, 16-bit | CC0 | `6f30a5d4fe38c928ded972704f085975f8dc0d65d9aa366df00e5a9d449fddd7` |
| BBBC039 masks | https://data.broadinstitute.org/bbbc/BBBC039/masks.zip | segmentation masks | 200 | 520×696 | CC0 | `f9e6043d8ca56344a4886f96a700d804d6ee982f31e2b2cd3194af2a053c2710` |
| AI4Life-MDC25 Nuclei | https://zenodo.org/records/15624741 (DOI 10.5281/zenodo.15624741) | real test / cross-source | 2,457 small + 20 large | 256×256 / 1024×1024 | CC BY 4.0 | `c7f2f5477a205e6020b74fded45d8f34f5383c82d261f5f401c1af58306dd7dc` |
| AI4Life-MDC25 FMD | https://zenodo.org/records/15761900 (DOI 10.5281/zenodo.15761900) | second real test | 36 of 301 held out | 256×256, float32 | CC BY 4.0 | `5110e958bef215e42649c2477855ea11f402842e11523ba0aebf05a634d7eb8f` |

## Splits

- Main split: FOV-level 56 train / 20 validation / 20 test, seed 0. No FOV is
  shared across splits. Validation is used for early stopping and for tuning
  all classical baselines. Test is untouched.
- External datasets are used only for zero-shot evaluation; the 20 external
  FOVs per dataset are spread through the sorted file list and were not used for
  training or tuning.
- BBBC014 channel 1 is the FITC/cytoplasm channel. BBBC015 and BBBC016 are U2OS
  cytoplasm-to-nucleus assays; BBBC015 files encode two channels and only
  channel 0 is used. BBBC018 provides a different cell type; its actin signal is
  in the green channel, which is used directly.
- BBBC039 uses the official 100/50/50 FOV split; masks are converted to a
  foreground mask (mask > 0) for Dice/IoU.
- AI4Life-MDC25 combines two sources: 2,457 256×256 pairs from the Jacquemet
  noisy-nuclei dataset and 20 1024×1024 pairs derived from BBBC006. After
  excluding 20 small-pair IDs that coincide numerically with the large-field
  IDs (a precaution; the images are not crops), the 2,437 small pairs are split
  by image ID into 1,951/243/243 train/validation/test. The test split is used
  only for the final comparison; all 243 test pairs are reported. The 20 large
  fields are a cross-source check and are never used for training or tuning.
- AI4Life-MDC25-FMD contains 63,669 256×256 noisy/clean pairs (multiple frames
  per field) from three microscopes (CF/WF/TP), three channels, and four noise
  levels. We extracted one frame from each of 1,998 fields (seed 0) and split by field, stratified by
  microscope/channel/level, into 1,396/301/301. The main comparison uses 36
  held-out fields, one per microscope/channel/level; no frame of a field is
  shared across splits.

## Forward model

For a clean image `x`, `y = Poisson(P * (g_sigma * x)) / P`, where
`g_sigma` is a normalized Gaussian PSF with `sigma = 2` pixels and `P` is the
full-scale photon count per pixel. Unless stated otherwise `P` is 3, 10, 30,
100, or 300. The same PSF is used to generate the data and by the TV/Wiener/RL baselines.
The images are normalized by their 8-bit full scale; `P` is therefore a
relative full-scale photon count, not a calibrated photons-per-emitter value.

## Download

```bash
mkdir -p datasets && cd datasets
for z in BBBC014/BBBC014_v1_images.zip BBBC015/BBBC015_v1_images.zip \
         BBBC016/BBBC016_v1_images.zip BBBC018/BBBC018_v1_images.zip; do
  curl -L -O https://data.broadinstitute.org/bbbc/$z
done
```

The archives extract into nested `BBBC0xx_v1_images/` directories; the loader
deduplicates by basename and expects this layout.
