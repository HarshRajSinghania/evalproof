"""Recompute every figure in MEASUREMENTS.md from the committed run files. No model needed.

    uv run python examples/support-qa/measure.py

The answer key (a regex per question, written before any model output was seen) stands
in for human review. "Truth" below means the answer key on all 83 cases — the
measurements also report where that key is itself wrong.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

from evalproof.calibration import acc_estimate, calibrate, ppi_estimate, raw_estimate
from evalproof.compare import compare_runs
from evalproof.config import load_config
from evalproof.gate import decide
from evalproof.report import calibration_markdown, comparison_markdown
from evalproof.runner import load_run
from evalproof.stats import percentile

HERE = Path(__file__).parent
SEEDS = 200
GOLD_N = 30


def verdicts(run, name):
    return {
        r.case_id: r.scores[name].passed for r in run.results if r.scores[name].passed is not None
    }


def sample_study(run, label: str) -> list[str]:
    """Draw a 30-case gold sample 200 times; how often does each estimate cover the truth?"""
    judge, key = verdicts(run, "judge"), verdicts(run, "answer_key")
    ids = sorted(set(judge) & set(key))
    truth = sum(key[c] for c in ids) / len(ids)
    rows = {"raw judge": [], "gold sample only": [], "ACC": [], "PPI": []}
    for seed in range(SEEDS):
        lab = sorted(random.Random(seed).sample(ids, GOLD_N))
        unl = [c for c in ids if c not in set(lab)]
        jl, gl = [judge[c] for c in lab], [key[c] for c in lab]
        ests = {
            "raw judge": raw_estimate([judge[c] for c in ids]),
            "gold sample only": raw_estimate(gl),  # same Wilson interval, on 30 labels
            "ACC": acc_estimate([judge[c] for c in ids], jl, gl, n_boot=2000, seed=seed),
            "PPI": ppi_estimate([judge[c] for c in unl], jl, gl),
        }
        for k, e in ests.items():
            rows[k].append(e)
    out = [
        f"#### {label}: {SEEDS} random gold samples of {GOLD_N} (truth = {100 * truth:.1f}%)",
        "",
        "| Estimate | Mean point | Mean \\|error\\| | 95% CI covers truth | Mean CI width "
        "| Undefined |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, es in rows.items():
        ok = [e for e in es if e.point is not None and e.low is not None and e.high is not None]
        undefined = len(es) - len(ok)
        mean_pt = statistics.mean(e.point for e in ok)
        mae = statistics.mean(abs(e.point - truth) for e in ok)
        cover = sum(e.low <= truth <= e.high for e in ok) / len(ok)
        width = statistics.mean(e.high - e.low for e in ok)
        out.append(
            f"| {k} | {100 * mean_pt:.1f}% | {100 * mae:.1f} pts | {100 * cover:.1f}% | "
            f"{100 * width:.1f} pts | {undefined} |"
        )
    return out


def latency(run) -> str:
    lat = [r.output.latency_ms for r in run.results if not r.output.error]
    toks = [r.output.tokens_out for r in run.results]
    return (
        f"system p50 {percentile(lat, 50):.0f} ms, p95 {percentile(lat, 95):.0f} ms, "
        f"mean output {statistics.mean(toks):.0f} tokens; "
        f"run wall time {run.info.wall_time_s:.0f} s"
    )


def main() -> None:
    cfg = load_config(HERE / "suite.yaml")
    base = load_run(HERE / "runs" / "baseline.json")
    cand = load_run(HERE / "runs" / "candidate.json")
    lines = ["## Calibration against the full answer key", ""]
    for label, run in (("baseline", base), ("candidate", cand)):
        rep = calibrate(run, "judge", gold_scorer="answer_key", seed=0)
        lines += [f"### {label}", "", calibration_markdown(rep)]
    lines += ["## Calibration with a 30-case gold sample (seed 0)", ""]
    for label, run in (("baseline", base), ("candidate", cand)):
        rep = calibrate(run, "judge", gold_scorer="answer_key", gold_sample=GOLD_N, seed=0)
        lines += [f"### {label}", "", calibration_markdown(rep)]
    lines += ["## Does the correction hold across samples?", ""]
    for label, run in (("baseline", base), ("candidate", cand)):
        lines += [*sample_study(run, label), ""]
    lines += ["## Baseline vs candidate", ""]
    cmp = compare_runs(base, cand)
    assert cfg.gate is not None
    lines += [comparison_markdown(cmp, decide(cmp, cfg.gate), cfg.gate.metric)]
    lines += ["## Latency", "", f"- baseline: {latency(base)}", f"- candidate: {latency(cand)}", ""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
