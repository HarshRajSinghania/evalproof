"""End to end: suite → run → compare → gate, with the offline example and a mocked Ollama judge."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest

import evalproof.cli as cli
from evalproof.cli import main
from evalproof.config import load_config
from evalproof.gate import EXIT_PASS, EXIT_REGRESSION, EXIT_USAGE
from evalproof.runner import load_run, run_suite, save_run
from tests.conftest import ScriptedProvider, mock_client

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "deterministic"


@pytest.fixture
def example(tmp_path: Path) -> Path:
    dst = tmp_path / "deterministic"
    shutil.copytree(
        EXAMPLE, dst, ignore=shutil.ignore_patterns("runs", ".evalproof", "__pycache__")
    )
    return dst


def _run(example: Path, variant: str, out: Path) -> int:
    return main(
        [
            "run",
            str(example / "suite.yaml"),
            "--system",
            f"python:router:{variant}_route",
            "--out",
            str(out),
            "--quiet",
        ]
    )


def test_offline_example_gate_exit_codes(example: Path, tmp_path: Path, capsys) -> None:
    runs = {v: tmp_path / f"{v}.json" for v in ("baseline", "regressed", "noisy", "refactored")}
    for variant, path in runs.items():
        assert _run(example, variant, path) == EXIT_PASS
    suite = str(example / "suite.yaml")
    expected = {"regressed": EXIT_REGRESSION, "noisy": EXIT_PASS, "refactored": EXIT_PASS}
    for variant, code in expected.items():
        argv = ["compare", str(runs["baseline"]), str(runs[variant]), "--suite", suite, "--gate"]
        assert main(argv) == code, variant
    out = capsys.readouterr().out
    assert "not distinguishable from noise" in out


def test_compare_writes_json_and_step_summary(example: Path, tmp_path: Path) -> None:
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    _run(example, "baseline", b)
    _run(example, "regressed", c)
    summary, payload = tmp_path / "summary.md", tmp_path / "gate.json"
    summary.write_text("# earlier step\n", "utf-8")
    code = main(
        [
            "compare",
            str(b),
            str(c),
            "--metric",
            "queue",
            "--min-cases",
            "30",
            "--gate",
            "--markdown",
            str(summary),
            "--json",
            str(payload),
        ]
    )
    assert code == EXIT_REGRESSION
    assert summary.read_text("utf-8").startswith("# earlier step\n")  # appended, not replaced
    data = json.loads(payload.read_text("utf-8"))
    assert data["gate"]["exit_code"] == EXIT_REGRESSION
    # Without --gate the same comparison reports but exits 0.
    assert main(["compare", str(b), str(c), "--metric", "queue", "--min-cases", "30"]) == EXIT_PASS


def test_compare_gate_needs_a_metric(example: Path, tmp_path: Path, capsys) -> None:
    b = tmp_path / "b.json"
    _run(example, "baseline", b)
    assert main(["compare", str(b), str(b), "--gate"]) == EXIT_USAGE
    assert "needs a gate" in capsys.readouterr().err
    # A run compared with itself is the canonical "nothing changed": the gate must pass.
    assert main(["compare", str(b), str(b), "--metric", "queue", "--gate"]) == EXIT_PASS


def test_report_and_errors(example: Path, tmp_path: Path, capsys) -> None:
    b = tmp_path / "b.json"
    _run(example, "baseline", b)
    assert main(["report", str(b)]) == EXIT_PASS
    assert "ticket-router" in capsys.readouterr().out
    assert main(["run", str(tmp_path / "missing.yaml")]) == EXIT_USAGE
    with pytest.raises(SystemExit):
        main(["--version"])


JUDGE_REPLY = '{"reason": "matches", "pass": %s, "score": %s}'


def _fake_ollama(req: httpx.Request) -> httpx.Response:
    if req.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.0.test"})
    if req.url.path == "/api/tags":
        return httpx.Response(
            200, json={"models": [{"name": "judge-m", "digest": "abc123def4567"}]}
        )
    body = json.loads(req.content)
    content = body["messages"][-1]["content"]
    answer = content.split("Response to grade:\n", 1)[-1]
    ok = "true" if "billing" in answer else "false"
    return httpx.Response(
        200,
        json={
            "message": {"content": JUDGE_REPLY % (ok, "1" if ok == "true" else "0")},
            "prompt_eval_count": 50,
            "eval_count": 12,
        },
    )


def test_run_suite_with_mocked_judge_and_cache(example: Path, tmp_path: Path) -> None:
    suite = example / "judged.yaml"
    suite.write_text(
        (example / "suite.yaml")
        .read_text("utf-8")
        .replace("gate:", "judge:\n  target: ollama:judge-m\ngate:")
        + "  # judged\n",
        "utf-8",
    )
    text = suite.read_text("utf-8").replace(
        "scorers:\n", "scorers:\n  - {name: judge, type: llm_judge}\n"
    )
    text += "calibration: {judge_scorer: judge, gold_scorer: queue, gold_sample: 20}\n"
    suite.write_text(text, "utf-8")
    cfg = load_config(suite)
    client = mock_client(_fake_ollama)
    seen: list[str] = []
    run = run_suite(cfg, client=client, progress=lambda i, n, r: seen.append(r.case_id))
    assert len(seen) == 60 and run.info.environment["ollama_version"] == "0.0.test"
    assert run.info.environment["digest:judge-m"] == "abc123def456"
    judged = [r.scores["judge"].passed for r in run.results]
    assert sum(judged) == sum(1 for r in run.results if r.output.output == "billing")
    # A second run is served from the cache: identical scores, no new HTTP chat calls.
    calls = {"n": 0}

    def counting(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/chat":
            calls["n"] += 1
        return _fake_ollama(req)

    again = run_suite(cfg, client=mock_client(counting))
    assert calls["n"] == 0
    assert [r.scores for r in again.results] == [r.scores for r in run.results]
    path = save_run(run, tmp_path / "judged.json")
    assert load_run(path).results[0].case_id == run.results[0].case_id
    assert (
        main(
            [
                "calibrate",
                str(path),
                "--judge-scorer",
                "judge",
                "--gold-scorer",
                "queue",
                "--gold-sample",
                "20",
                "--json",
                str(tmp_path / "cal.json"),
            ]
        )
        == EXIT_PASS
    )
    assert json.loads((tmp_path / "cal.json").read_text("utf-8"))["n_gold"] == 20


def test_run_suite_concurrency_and_limit(example: Path) -> None:
    cfg = load_config(example / "suite.yaml").model_copy(update={"concurrency": 4})
    run = run_suite(cfg, limit=10, use_cache=False)
    assert [r.case_id for r in run.results] == [f"t{i:02d}" for i in range(1, 11)]


def test_calibrate_with_labels_file(example: Path, tmp_path: Path) -> None:
    b = tmp_path / "b.json"
    _run(example, "baseline", b)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(json.dumps({"id": f"t{i:02d}", "label": i % 2 == 0}) for i in range(1, 21)),
        "utf-8",
    )
    assert (
        main(["calibrate", str(b), "--judge-scorer", "queue", "--labels", str(labels)]) == EXIT_PASS
    )


def test_pairwise_command(example: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    suite = example / "suite.yaml"
    suite.write_text(
        suite.read_text("utf-8").replace("gate:", "judge:\n  target: ollama:judge-m\ngate:"),
        "utf-8",
    )
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    _run(example, "baseline", b)
    _run(example, "regressed", c)
    monkeypatch.setattr(cli, "make_provider", lambda cfg: ScriptedProvider(['{"winner": "A"}']))
    code = main(
        [
            "pairwise",
            str(b),
            str(c),
            "--suite",
            str(suite),
            "--limit",
            "5",
            "--no-cache",
            "--json",
            str(tmp_path / "pw.json"),
        ]
    )
    assert code == EXIT_PASS
    out = capsys.readouterr().out
    assert "Position-inconsistent verdicts" in out and "5/5" in out
    assert len(json.loads((tmp_path / "pw.json").read_text("utf-8"))) == 5
