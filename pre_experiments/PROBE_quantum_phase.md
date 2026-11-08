# Pre-experiment: near-optimality of adaptive quantum phase estimation

**Question.** Is there headroom for a learned (RL) policy in adaptive quantum
phase estimation, or is a Bayesian adaptive estimator already near the quantum
Cramér-Rao / standard quantum limit (SQL, MSE ~ 1/N)?

## Model

- Two-level probe, one photon per measurement, phase φ uniform on [0, 2π).
- Each measurement chooses a basis θ; outcome is 0 with probability
  cos²((φ−θ)/2), otherwise 1.
- Policies:
  - fixed: θ = 0;
  - random: θ drawn uniformly from 9 bases;
  - adaptive: discretized Bayesian posterior on 2000 grid points; at each step
    choose θ minimizing the expected posterior variance after the outcome.
- 60 random true phases per point; circular MSE.

## Results (MSE, mean over trials)

| N | fixed | random | adaptive | SQL = 1/N | adaptive / SQL |
|---|---|---|---|---|---|
| 10 | 1.363 | 0.166 | 0.189 | 0.100 | 1.89 |
| 20 | 0.861 | 0.070 | 0.062 | 0.050 | 1.24 |
| 50 | 0.904 | 0.026 | 0.020 | 0.020 | 0.97 |
| 100 | 0.667 | 0.009 | 0.009 | 0.010 | 0.89 |
| 200 | 0.845 | 0.006 | 0.008 | 0.005 | 1.53 |

## Interpretation

The Bayesian adaptive policy is within a factor of ~0.9–1.9 of the SQL across
N = 10–200. A learned policy would have to beat a near-CRLB estimator; the
headroom is small. Per the near-optimality rule, an RL-for-quantum-sensing paper
on this task is not justified.

## Caveat

This is a toy model without decoherence, readout noise, or hardware constraints.
A real experiment could differ, but the burden would be on the new method to
show a gain over the Bayesian baseline; the pre-experiment gives no reason to
expect one.

## Reproduction

```bash
python pre_experiments/quantum_phase_probe.py
```

Result: `pre_experiments/quantum_phase_probe.json`.
