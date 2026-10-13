"""Retrieval metrics over ``SystemOutput.retrieved`` against a case's ``relevant`` ids."""

from __future__ import annotations

import math

from evalproof.models import Case, Score, SystemOutput


def _relevance(case: Case, field: str) -> dict[str, float]:
    value = case.field(field)
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    return {str(v): 1.0 for v in value}


class _RetrievalScorer:
    def __init__(
        self, name: str, k: int = 10, field: str = "relevant", pass_threshold: float | None = None
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.name, self.k, self.field, self.pass_threshold = name, k, field, pass_threshold

    def _finish(self, value: float) -> Score:
        passed = None if self.pass_threshold is None else value >= self.pass_threshold
        return Score(value=value, passed=passed)

    def metric(self, rel: dict[str, float], ranked: list[str]) -> float:
        raise NotImplementedError

    def score(self, case: Case, out: SystemOutput) -> Score:
        rel = _relevance(case, self.field)
        if not rel:
            return Score(value=None, reason=f"case has no {self.field!r}")
        return self._finish(self.metric(rel, out.retrieved[: self.k]))


class RecallAtK(_RetrievalScorer):
    def metric(self, rel: dict[str, float], ranked: list[str]) -> float:
        relevant = {d for d, g in rel.items() if g > 0}
        return len(relevant & set(ranked)) / len(relevant) if relevant else 0.0


class MRR(_RetrievalScorer):
    def metric(self, rel: dict[str, float], ranked: list[str]) -> float:
        for i, doc in enumerate(ranked, start=1):
            if rel.get(doc, 0) > 0:
                return 1.0 / i
        return 0.0


class NDCG(_RetrievalScorer):
    """nDCG@k with exponential gain ``2^rel - 1``."""

    def metric(self, rel: dict[str, float], ranked: list[str]) -> float:
        def dcg(gains: list[float]) -> float:
            return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))

        actual = dcg([rel.get(doc, 0.0) for doc in ranked])
        ideal = dcg(sorted(rel.values(), reverse=True)[: self.k])
        return actual / ideal if ideal > 0 else 0.0
