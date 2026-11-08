# Pre-experiment: task/dose identifiability (LLM/soft-token conditioning gate)

**Question.** In multi-task microscopy restoration, is explicit task/language
conditioning likely to add information, or is the blind model already
near-optimal because the acquisition task and dose are encoded in the image?

## Method

- Four tasks: BBBC014 channel 1, BBBC015 channel 0, BBBC016 GFP, BBBC018 actin.
- 10 train / 10 test FOVs per task (FOV-level split), 20 patches per FOV,
  $128\times128$, Poisson-blurred at $P=30$ with the main forward model.
- Linear probe (logistic regression) on 14 simple features: intensity moments,
  percentiles, gradient energy, Laplacian variance, and three radial FFT bands.
- Small CNN probe (3 conv blocks, global average pool, 12 epochs).
- Dose probe: BBBC014, five doses $\{3,10,30,100,300\}$, 10/10 FOVs, 20
  patches per FOV.

## Results

| Probe | Linear | CNN | n train / test |
|---|---|---|---|
| Task (4 classes) | 0.795 | 0.757 | 800 / 800 |
| Dose (5 classes) | 0.979 | 1.000 | 1000 / 1000 |

Chance levels are 0.25 (task) and 0.20 (dose).

## Interpretation

- **Dose is perfectly identifiable** from the noisy image. Explicit dose
  conditioning cannot add information; this is consistent with the measured
  0.15--0.22 dB gap between the blind and dose-conditioned models in the main
  study. The blind model already infers the dose internally.
- **Task is ~80% identifiable** from simple features. The blind model therefore
  already has most of the task information; only the ambiguous ~20% could
  benefit from an explicit task label or a language/soft-token embedding.
- Per the near-optimality rule, a full LLM/soft-token conditioning paper is not
  justified: the expected restoration gain is bounded by a small ambiguous
  subset, while the blind model already provides a positive, simpler result.

## Caveat

Probe accuracy is not the same as a restoration gain. A conditioned model could
still help on the ambiguous cases. This is a go/no-go heuristic, not a proof;
the full conditioning experiment was not run because the expected effect is
small relative to the cost.

## Reproduction

```bash
python src/probe_task_dose.py --dose 30 --n-patch 20 \
  --out results/probe_task_dose.json
```

Result: `results/probe_task_dose.json` (copy in this directory).
