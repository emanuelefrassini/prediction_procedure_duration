"""Artificial phase trajectories and whole-procedure train/validation/test folds."""
import numpy as np
import torch

N_PHASES = 14


def synthetic_cases(n=90, seed=42, interval_seconds=30):
    if n < 6 or interval_seconds <= 0:
        raise ValueError('require at least six cases and positive sampling interval')
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        # Variable dwell times, with a latent procedure-wide speed factor.
        speed = rng.lognormal(0, .27)
        durations = np.maximum(1, rng.poisson(np.linspace(2, 5, N_PHASES) * speed))
        phases = np.repeat(np.arange(N_PHASES), durations).astype(np.int64)
        remaining = (len(phases) - 1 - np.arange(len(phases))) * interval_seconds
        cases.append((phases, remaining.astype(np.float32)))
    return cases


def folds(n, count=3, seed=42):
    if not 2 <= count <= n or n - int(np.ceil(n / count)) < 2:
        raise ValueError('need at least two cases in each outer training pool')
    parts = np.array_split(np.random.default_rng(seed).permutation(n), count)
    for i, test in enumerate(parts):
        pool = np.concatenate([p for j, p in enumerate(parts) if j != i])
        pool = np.random.default_rng(seed + 1009 + i).permutation(pool)
        nv = max(1, min(len(pool)-1, int(np.ceil(.2*len(pool)))))
        yield pool[nv:], pool[:nv], test


def collate(cases):
    """Pad right with -1; lengths are never passed as model inputs."""
    n = len(cases)
    t = max(len(p) for p, _ in cases)
    x = torch.full((n,t), -1, dtype=torch.long)
    y = torch.full((n,t), -1., dtype=torch.float32)
    for i,(p,r) in enumerate(cases):
        x[i,:len(p)] = torch.as_tensor(p)
        y[i,:len(r)] = torch.as_tensor(r)
    return x,y


def baseline_fit(cases):
    """Train-only average remaining time for each phase; no test information."""
    phase = np.concatenate([p for p, _ in cases])
    target = np.concatenate([r for _, r in cases])
    return np.array([target[phase==i].mean() if np.any(phase==i) else target.mean()
                     for i in range(N_PHASES)])
