from __future__ import annotations

import pytest

from evalproof.calibration import (
    Confusion,
    acc_estimate,
    acc_point,
    calibrate,
    confusion,
    ppi_estimate,
    raw_estimate,
)
from evalproof.stats import z_for
from tests.conftest import make_run


def test_confusion_and_kappa_known_values() -> None:
    # tp=40 fp=10 fn=5 tn=45 → TPR 40/45, FPR 10/55, acc .85,
    # pe = (50·45 + 50·55)/100² = .5 → κ = (.85-.5)/(1-.5) = .7
    cm = Confusion(tp=40, fp=10, fn=5, tn=45)
    assert cm.n == 100
    assert cm.tpr == pytest.approx(40 / 45)
    assert cm.fpr == pytest.approx(10 / 55)
    assert cm.accuracy == pytest.approx(0.85)
    assert cm.kappa == pytest.approx(0.7)


def test_confusion_degenerate() -> None:
    empty = Confusion(0, 0, 0, 0)
    assert (
        empty.tpr is None and empty.fpr is None and empty.accuracy is None and empty.kappa is None
    )
    assert Confusion(tp=5, fp=0, fn=0, tn=0).kappa is None  # pe == 1


def test_confusion_from_lists() -> None:
    cm = confusion([True, True, False, False], [True, False, True, False])
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (1, 1, 1, 1)
    with pytest.raises(ValueError):
        confusion([True], [True, False])


def test_acc_point_known_values() -> None:
    assert acc_point(0.5, tpr=0.9, fpr=0.1) == pytest.approx(0.5)
    assert acc_point(0.58, tpr=0.9, fpr=0.1) == pytest.approx(0.6)
    assert acc_point(0.95, tpr=0.9, fpr=0.1) == 1.0  # clipped
    assert acc_point(0.5, tpr=0.4, fpr=0.4) is None  # judge no better than chance


def test_ppi_known_values() -> None:
    # f_u mean .6, var(ddof=1) = 10/9·.24 = .266667; resid = y - f_l has mean -.1 and
    # var(ddof=1) = (9·.01 + .81)/9 = .1 → θ = .5, se = √(.026667+.01) = .191485
    f_u = [True] * 6 + [False] * 4
    f_l = [True] * 5 + [False] * 5
    y = [True] * 4 + [False] * 6
    est = ppi_estimate(f_u, f_l, y)
    half = z_for(0.05) * 0.191485
    assert est.point == pytest.approx(0.5)
    assert est.low == pytest.approx(0.5 - half, abs=1e-5)
    assert est.high == pytest.approx(0.5 + half, abs=1e-5)
    assert est.n == 20


def test_ppi_needs_both_sets() -> None:
    assert ppi_estimate([], [True], [True]).point is None


def test_raw_estimate_is_wilson() -> None:
    est = raw_estimate([True] * 6 + [False] * 4)
    assert est.point == pytest.approx(0.6)
    assert est.low is not None and est.low < 0.6 < (est.high or 0)


def _synthetic(n_pos: int = 100, n_neg: int = 100, tp: int = 90, fp: int = 30):
    """Gold rate .5; judge TPR .9, FPR .3 → raw rate .6, ACC exactly .5."""
    outcomes: dict[str, dict[str, bool | float | None]] = {}
    for i in range(n_pos):
        outcomes[f"p{i:03d}"] = {"judge": i < tp, "gold": True}
    for i in range(n_neg):
        outcomes[f"n{i:03d}"] = {"judge": i < fp, "gold": False}
    return make_run(outcomes)


def test_acc_recovers_truth_with_full_gold() -> None:
    run = _synthetic()
    judge = [r.scores["judge"].passed for r in run.results]
    gold = [r.scores["gold"].passed for r in run.results]
    est = acc_estimate(judge, judge, gold, n_boot=500)  # type: ignore[arg-type]
    assert est.point == pytest.approx(0.5)
    assert est.low is not None and est.low < 0.5 < (est.high or 0)


def test_acc_single_class_gold() -> None:
    est = acc_estimate([True, False], [True, False], [True, True])
    assert est.point is None and "one class" in est.note


def test_acc_chance_judge() -> None:
    est = acc_estimate([True, False, True, False], [True, False], [True, False][::-1])
    assert est.point is None


def test_calibrate_from_gold_scorer_sample() -> None:
    run = _synthetic()
    rep = calibrate(run, "judge", gold_scorer="gold", gold_sample=80, seed=3, n_boot=500)
    assert rep.n_judged == 200 and rep.n_gold == 80
    assert rep.truth is not None and rep.truth.point == pytest.approx(0.5)
    by = {e.method: e for e in rep.estimates}
    assert by["raw judge pass rate"].point == pytest.approx(0.6)
    # The raw rate is biased (.6) and its CI excludes the truth; both corrections cover it.
    assert (by["raw judge pass rate"].low or 0) > 0.5
    for method in ("ACC", "PPI"):
        e = by[method]
        assert e.low is not None and e.high is not None and e.low <= 0.5 <= e.high
    assert rep.disagreements  # FP/FN in the sample are listed


def test_calibrate_with_labels_and_undecided() -> None:
    run = make_run(
        {
            "a": {"judge": True},
            "b": {"judge": False},
            "c": {"judge": None},
            "d": {"judge": True},
            "e": {"judge": False},
        }
    )
    rep = calibrate(run, "judge", labels={"a": True, "b": True, "d": False, "zz": True})
    assert rep.undecided_judge == 1
    assert rep.n_gold == 3
    assert rep.truth is None
    assert sorted(rep.disagreements) == ["b", "d"]


def test_calibrate_argument_errors() -> None:
    run = _synthetic()
    with pytest.raises(ValueError):
        calibrate(run, "judge")
    with pytest.raises(ValueError):
        calibrate(run, "judge", labels={}, gold_scorer="gold")
    with pytest.raises(ValueError):
        calibrate(run, "missing", gold_scorer="gold")
    with pytest.raises(ValueError):
        calibrate(run, "judge", labels={"nope": True})
