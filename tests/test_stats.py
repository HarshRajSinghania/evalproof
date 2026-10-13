"""Known-value tests: every expected number below is worked by hand in the comment."""

from __future__ import annotations

import math

import pytest

from evalproof.stats import (
    binom_cdf,
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap,
    percentile,
    wilson,
    z_for,
)


def test_z_for_95() -> None:
    assert z_for(0.05) == pytest.approx(1.959964, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_z_for_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        z_for(alpha)


def test_wilson_zero_of_ten() -> None:
    # z²=3.8415; denom=1.38415; centre=(3.8415/20)/1.38415=0.13876; half=0.13876
    lo, hi = wilson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.27753, abs=1e-4)


def test_wilson_half() -> None:
    # p=.5, n=10: centre .5, half = 1.96*sqrt(.025+.0096037)/1.38415 = .26336
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.23659, abs=1e-4)
    assert hi == pytest.approx(0.76341, abs=1e-4)


def test_wilson_degenerate_and_invalid() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_binom_cdf() -> None:
    # P(X<=2), X~Bin(10,.5) = (1+10+45)/1024
    assert binom_cdf(2, 10) == pytest.approx(56 / 1024)


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 5, 2 / 32),  # 2 * .5^5
        (2, 8, 2 * 56 / 1024),  # 2 * P(X<=2 | n=10)
        (5, 5, 1.0),  # symmetric → capped at 1
        (0, 0, 1.0),  # no discordant pairs, no evidence
    ],
)
def test_mcnemar_exact(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(expected)


def test_mcnemar_rejects_negative() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_paired_bootstrap_identical_runs() -> None:
    res = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1])
    assert res.delta == 0.0
    assert res.p_value == 1.0
    assert (res.ci_low, res.ci_high) == (0.0, 0.0)


def test_paired_bootstrap_total_regression() -> None:
    res = paired_bootstrap([1.0] * 30, [0.0] * 30)
    assert res.delta == -1.0
    assert res.ci_low == res.ci_high == -1.0
    assert res.p_value == 0.0
    assert res.n == 30


def test_paired_bootstrap_is_seeded() -> None:
    a = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    b = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0]
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_paired_bootstrap_validates() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])


def test_bootstrap_ci_brackets_mean() -> None:
    values = [float(i) for i in range(100)]
    lo, hi = bootstrap_ci(values, seed=1)
    assert lo < 49.5 < hi
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_percentile() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert math.isclose(percentile([0.0, 100.0], 95), 95.0)
