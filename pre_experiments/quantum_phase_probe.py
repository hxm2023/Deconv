"""Near-optimality gate for RL in adaptive quantum phase estimation.

A discretized Bayesian adaptive policy is compared with the standard quantum
limit (SQL, MSE ~ 1/N) for a two-level probe with one photon per measurement.
If the Bayesian policy is already within a small factor of the bound, a learned
policy has little headroom.
"""
from __future__ import annotations

import json

import numpy as np

GRID = np.linspace(0.0, 2 * np.pi, 2001)[:-1]
THETAS = np.linspace(0.0, np.pi, 9)


def run(phi_true, n_photons, policy, rng):
    post = np.ones_like(GRID) / len(GRID)
    for _ in range(n_photons):
        if policy == "fixed":
            theta = 0.0
        elif policy == "random":
            theta = float(rng.choice(THETAS))
        else:
            best = None
            for th in THETAS:
                p0 = np.cos((GRID - th) / 2.0) ** 2
                ev = 0.0
                for w in (post * p0, post * (1.0 - p0)):
                    z = w.sum()
                    if z < 1e-12:
                        continue
                    mu = (w * GRID).sum() / z
                    ev += z * (w * (GRID - mu) ** 2).sum() / z
                if best is None or ev < best[0]:
                    best = (ev, th)
            theta = best[1]
        p0 = np.cos((phi_true - theta) / 2.0) ** 2
        outcome = rng.random() < p0
        post = post * (np.cos((GRID - theta) / 2.0) ** 2 if outcome
                       else np.sin((GRID - theta) / 2.0) ** 2)
        post /= post.sum()
    mean = np.angle((post * np.exp(1j * GRID)).sum())
    return float(np.angle(np.exp(1j * (mean - phi_true))) ** 2)


def main():
    rng = np.random.default_rng(0)
    ns = [10, 20, 50, 100, 200]
    trials = 60
    out = {}
    for n in ns:
        row = {}
        for policy in ("fixed", "random", "adaptive"):
            mses = [run(rng.uniform(0, 2 * np.pi), n, policy, rng)
                    for _ in range(trials)]
            row[policy] = float(np.mean(mses))
        row["sql"] = 1.0 / n
        row["ratio_adaptive"] = row["adaptive"] / row["sql"]
        out[str(n)] = row
        print(n, {k: round(v, 4) for k, v in row.items()}, flush=True)
    with open("pre_experiments/quantum_phase_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote pre_experiments/quantum_phase_probe.json")


if __name__ == "__main__":
    main()
