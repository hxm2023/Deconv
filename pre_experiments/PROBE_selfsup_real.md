# Pre-experiment: label-free self-supervised real denoising

**Gate.** Train on real noisy images only (no clean targets) and beat tuned
BM3D (20.92 dB / 0.348 SSIM) on the 40 held-out AI4Life pairs.

## Method

- J-invariant masked-pixel U-Net: the input is the noisy image multiplied by a
  binary mask; the loss is MSE on the masked pixels. At test, two complementary
  masks are used and each pixel is taken from the prediction where it was
  masked.
- Masks: checkerboard (parity), random 4x4 blocks, random 8x8 blocks.
- Test-time averaging: 8 random masks per image.
- Capacity/scale: U-Net 32 channels, 8k patches, 40 epochs (default); one run
  with 48 channels, 16k patches, 80 epochs.
- Training data: the same 1,951 real noisy small pairs; the clean references
  are never used for training or model selection.

## Results (40 held-out pairs, mean)

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Noisy input | 16.10 | 0.104 |
| Checkerboard J-invariant | 16.10 | 0.104 |
| Block-4 | **20.56** | 0.356 |
| Block-4, 8-mask average | 20.39 | **0.362** |
| Block-8 | 19.91 | 0.348 |
| Block-4, ch48, 16k/80 | 17.69 | 0.194 |
| Tuned BM3D | 20.92 | 0.348 |
| Supervised real U-Net | 28.00 | 0.839 |

## Interpretation

- The checkerboard mask fails because the real camera noise is spatially
  correlated: the network can predict the masked noisy value from its
  neighbours and learns the identity (output equals the input).
- A 4x4 block mask breaks the local noise correlation and learns real
  denoising: +4.46 dB over the noisy input without any clean target.
- The best self-supervised model is still 0.36 dB below tuned BM3D. A larger,
  longer-trained model overfits the masked-pixel objective (17.69 dB).
- The gate fails: label-free self-supervision approaches but does not beat the
  classical baseline. The supervised real model (28.00 dB) remains the positive
  real-data result; clean pairs are still worth their cost.

## Caveat

A better blind-spot architecture (dilated blind-spot, longer training with
early stopping on a proxy) might close the 0.36 dB gap. The pre-experiment gives
no evidence of a robust gain over BM3D, so this direction was not pursued.

## Reproduction

```bash
python src/train_real_selfsup.py --patches 8000 --epochs 40 --block 4 \
  --out checkpoints/real_selfsup_block4.pt
python src/eval_real_selfsup.py --ckpt checkpoints/real_selfsup_block4.pt \
  --block 4 --n-mask 2 --out results/real_selfsup_block4_test40.json
```

Results are copied in this directory.
