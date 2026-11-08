# Reproducing the paper

## Environment

Python 3.10+, `torch` with CUDA, `numpy`, `scipy`, `scikit-image`,
`matplotlib`, `pillow`, `scipy.stats`. Install with:

```bash
python -m pip install -r requirements.txt
```

Training/evaluation used an NVIDIA A100-PCIE-40GB. TV deconvolution and the CPU
latency measurement used one CPU core. The main random seed is 0; the test
noise seed is 2026.

## 1. Data

Follow `DATASETS.md`. Expected layout:

```
datasets/BBBC014_v1_images/BBBC014_v1_images/Channel 1-*.Bmp
datasets/BBBC015_v1_images/BBBC015_v1_images/*.JPG
datasets/BBBC016_v1_images/BBBC016_v1_images/*d0.TIF
datasets/BBBC018_v1_images/BBBC018_v1_images/*-actin.DIB
datasets/BBBC039_images/*.tif
datasets/masks_039/masks/*.png
datasets/meta_039/metadata/*.txt
datasets/real_nuclei/small_images/{noisy,gt}/*.tif
datasets/real_nuclei/large_images/{noisy,gt}/*.tif
```

For BBBC039, extract `images.zip` into `datasets/BBBC039_images`,
`masks.zip` into `datasets/masks_039`, and `metadata.zip` into
`datasets/meta_039`. The loader expects this layout.

## 2. Train

```bash
# main mixed-dose model (no dose input; 64 channels, 48k patches, 150 epochs)
python src/train_unet.py --dataset bbbc014 --n-val 20 --n-test 20 \
  --ch 64 --patches 48000 --epochs 150 --val-patches 4 --mixed-doses \
  --val-all-doses --out checkpoints/bbbc014_blind_strong.pt

# base-budget mixed-dose model used for the matched ablation
python src/train_unet.py --dataset bbbc014 --n-val 20 --n-test 20 \
  --patches 24000 --epochs 100 --val-patches 4 --mixed-doses \
  --val-all-doses --out checkpoints/bbbc014_blind3.pt

# conditioned ensemble seed 0 (uncertainty analysis)
python src/train_unet.py --dataset bbbc014 --n-val 20 --n-test 20 \
  --patches 24000 --epochs 100 --dose-cond --val-patches 4 --seed 0 \
  --out checkpoints/bbbc014_dcond.pt

# two more ensemble members (seeds 1 and 2)
python src/train_unet.py --dataset bbbc014 --n-val 20 --n-test 20 \
  --patches 24000 --epochs 100 --dose-cond --val-patches 4 --seed 1 \
  --out checkpoints/bbbc014_dcond_s1.pt
python src/train_unet.py --dataset bbbc014 --n-val 20 --n-test 20 \
  --patches 24000 --epochs 100 --dose-cond --val-patches 4 --seed 2 \
  --out checkpoints/bbbc014_dcond_s2.pt

# per-dose specialists (16k patches, 60-epoch schedule)
python src/train_unet.py --dataset bbbc014 --patches 16000 --epochs 60 \
  --photons 3 --val-patches 4 --out checkpoints/bbbc014_spec3.pt
python src/train_unet.py --dataset bbbc014 --patches 16000 --epochs 60 \
  --photons 30 --val-patches 4 --out checkpoints/bbbc014_spec30b.pt
python src/train_unet.py --dataset bbbc014 --patches 16000 --epochs 60 \
  --photons 100 --val-patches 4 --out checkpoints/bbbc014_spec100.pt

# BBBC039 restoration model for downstream segmentation
python src/train_seg.py --mode rest --patches 12000 --epochs 40   --batch 128 --out checkpoints/seg_rest_full.pt

# phase-scrambled negative control (equal 24k/100 budget)
python src/train_unet.py --dataset bbbc014 --patches 24000 --epochs 100 \
  --dose-cond --scramble --val-patches 4 --out checkpoints/bbbc014_scr2.pt

# budget-matched strong dose-conditioned ablation (64 channels, 48k, 150)
python src/train_unet.py --dataset bbbc014 --n-val 20 --n-test 20 \
  --ch 64 --patches 48000 --epochs 150 --dose-cond --val-patches 4 \
  --out checkpoints/bbbc014_dcond_strong.pt

# real noisy/clean model (AI4Life-MDC25 small pairs; 48 channels, 16k, 80)
python src/train_real.py --ch 48 --patches 16000 --epochs 80 --patch 128 \
  --out checkpoints/real_unet_strong.pt

# budget-matched DnCNN baseline (same 16k patches, 80 epochs)
python src/train_real_dncnn.py --ch 64 --patches 16000 --epochs 80 \
  --out checkpoints/real_dncnn_strong.pt

# PSF-mixed simulation model (50/50 Gaussian/Airy), same 64ch/48k/150 budget
python src/train_unet.py --dataset bbbc014 --n-val 20 --n-test 20 \
  --ch 64 --patches 48000 --epochs 150 --mixed-doses --val-all-doses \
  --psf-mix --val-patches 4 --out checkpoints/bbbc014_blind_psfmix.pt
```

Early stopping selects the best validation checkpoint; the selected epochs are
recorded in the checkpoints and logs.

## 3. Evaluate

```bash
# dose response of the conditioned model
python src/eval_unet_doses.py --ckpt checkpoints/bbbc014_dcond.pt \
  --doses 3,10,30,100,300 --n-val 20 --n-test 20 \
  --out results/unet_dose_response.json

# matched ablation: specialist/blind/scrambled/conditioned on the same patches
python src/eval_matched_ablation.py

# strong model dose response, budget-matched conditioning, and controls
python src/eval_blind_strong.py
python src/eval_dcond_strong.py
python src/eval_blind_controls.py

# fair TV-deconvolution baseline and protocol
python src/bench_tvdeconv.py
python src/tvdeconv_full.py          # optional 80-patch run (slow)
python src/make_fair_assets.py

# cross-dataset zero-shot table (TV deconv versus U-Net)
python src/bench_tvdeconv_external.py  # or the inline command in REVIEW_LOG

# real noisy/clean validation (tune classical on val, then held-out test)
python src/tune_real_classical.py --n-val 5 --workers 8
python src/eval_real_small_par.py --n 0 --workers 12
python src/eval_real_affine.py --n-test 243 --n-val 5 --workers 8 \
  --ckpt checkpoints/real_unet.pt --ch 32 \
  --out results/real_affine_test_all.json
python src/eval_real_unet_only.py --n-test 243 --workers 8 \
  --ckpt checkpoints/real_unet_strong.pt --ch 48 \
  --base results/real_affine_test_all.json \
  --out results/real_affine_test_all_strong.json
python src/eval_read_noise.py  # SI S4b
python src/eval_real_dncnn.py --ckpt checkpoints/real_dncnn_strong.pt --n 0 \
  --out results/real_dncnn_strong.json
python src/eval_psf_airy.py --ckpt checkpoints/bbbc014_blind_strong.pt \
  --out results/psf_airy.json       # tunes TV with the correct Airy PSF
python src/eval_psf_airy.py --ckpt checkpoints/bbbc014_blind_psfmix.pt \
  --params results/psf_airy.json --out results/psf_airy_mix.json
python src/eval_real_large_par.py --n-patch 8 --workers 12 \
  --ckpt checkpoints/real_unet_strong.pt --ch 48 \
  --out results/real_large_fair_strong.json  # cross-source check

# label-free self-supervised control (SI S7)
python src/train_real_selfsup.py --patches 8000 --epochs 40 --block 4 \
  --out checkpoints/real_selfsup_block4.pt
python src/eval_real_selfsup.py --ckpt checkpoints/real_selfsup_block4.pt \
  --block 4 --n-mask 2 --n 0 --out results/real_selfsup_all.json

# second real source: FMD (download Archive.zip from Zenodo 15761900 first)
python src/fmd_prepare.py --fields 2000 --out datasets/fmd
python src/fmd_resplit.py
python src/train_fmd.py --patches 8000 --epochs 40 --out checkpoints/fmd_unet.pt
python src/make_fmd_subset.py
python src/tune_fmd_micro.py
python src/tune_fmd_wf_extra.py   # extend WF BM3D grid to sigma 20
python src/eval_fmd_par.py --subset prereg/fmd_subset150.json \
  --params-micro results/fmd_tuning_micro.json --workers 8 \
  --out results/fmd_test150_micro.json
python src/analyze_fmd150.py results/fmd_test150_micro.json
python src/make_fmd_assets.py

# dose/task identifiability probes (SI S8)
python src/probe_task_dose.py --dose 30 --n-patch 20 \
  --out results/probe_task_dose.json
```

## 4. Figures and paper

```bash
python src/make_fig1.py
python src/make_fig2.py
python src/make_fig3.py
python src/make_fig4.py
python src/make_fig_seg.py
python src/make_fair_assets.py
python src/make_seg_numbers.py
python src/make_real_assets.py
python src/make_fig_real.py \
  --affine results/real_affine_test_all_strong.json \
  --ckpt checkpoints/real_unet_strong.pt --ch 48
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex si && pdflatex si
# main.pdf is the submission source; main_elsevier.tex is archival.
pdflatex si && pdflatex si
```

## 5. Key result files

- `results/tvdeconv_doses.json` — fair TV-deconv/TV-denoise/U-Net per-FOV results
- `results/matched_ablation.json` — specialist/blind/scrambled/conditioned
- `results/tvdeconv_external30.json` — cross-dataset TV deconv vs U-Net
- `results/uncertainty_fov.json` — ensemble uncertainty, PSNR, RMSE
- `results/adaptive_policy_analysis.json` — adaptive negative result
- `results/blind_fair_doses.json`, `results/scr2_matched.json` — main model and control
- `results/seg_doses_full.json` — BBBC039 Dice versus dose
- `results/real_small_test_all.json` — held-out real noisy/clean comparison (243 pairs)
- `results/real_small_test40.json` — first-40 subset (historical)
- `results/real_small_tuning.json` — classical parameters tuned on real validation pairs
- `results/real_large_fair.json` — cross-source large-field check
- `results/real_selfsup_all.json` — label-free self-supervised control
- `results/real_leakage_full.json`, `results/real_data_checks.json` — leakage and ground-truth checks
- `results/protocol_ci.json`, `results/uncertainty_ci.json` — bootstrap CIs

## 6. Notes

The dose sweep is simulated from high-SNR public references. AI4Life-MDC25
supplies real noisy/clean pairs for an in-source restoration check; these are
not calibrated low-dose measurements under a controlled photon budget, and the
large BBBC006 source is a cross-source negative check. The `P` values are
full-scale Poisson units, not calibrated photons per emitter. No test FOV was
used for training, early stopping, baseline tuning, target selection, or figure
selection.
