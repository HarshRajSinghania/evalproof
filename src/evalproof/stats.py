"""Small, readable statistics. numpy and the standard library only — no SciPy.

Every function here has a known-value test in ``tests/test_stats.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def z_for(alpha: float) -> float:
    """Two-sided critical value: ``z_for(0.05)`` ≈ 1.959964."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return NormalDist().inv_cdf(1 - alpha / 2)


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Behaves at 0/n and n/n."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = z_for(alpha)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Sequence[float] | FloatArray,
    stat: Callable[[FloatArray], float] = lambda x: float(np.mean(x)),
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval, seeded so a report is reproducible."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = np.array([stat(arr[row]) for row in idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


@dataclass(frozen=True)
class PairedDelta:
    """``delta = mean(candidate) - mean(baseline)`` over paired cases."""

    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def paired_bootstrap(
    baseline: Sequence[float] | FloatArray,
    candidate: Sequence[float] | FloatArray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDelta:
    """Resample *case indices*, so each replicate keeps baseline and candidate paired.

    The two-sided p-value is the percentile-bootstrap approximation
    ``2 * min(P(d* <= 0), P(d* >= 0))`` — adequate for gating, and stated as such.
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("need two non-empty arrays of equal length")
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    reps = diff[idx].mean(axis=1)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    if np.all(diff == 0):
        p = 1.0
    else:
        p = float(min(1.0, 2 * min(np.mean(reps <= 0), np.mean(reps >= 0))))
    return PairedDelta(float(diff.mean()), float(lo), float(hi), p, int(diff.size))


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs.

    ``b`` = baseline passed, candidate failed; ``c`` = baseline failed, candidate passed.
    Only discordant pairs carry information about a difference between the two systems.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_cdf(min(b, c), n))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
