# Contributing

```bash
uv sync
uv run ruff format . && uv run ruff check . && uv run mypy src
uv run pytest --cov
```

* Statistical code needs a known-value test: work the expected number out by hand and put
  the arithmetic in the test's comment (see `tests/test_stats.py`).
* Provider code is tested against `httpx.MockTransport`; tests never call a real model.
* A change to the gate's behaviour needs an ADR in `docs/adr/` and a case in the offline
  example that shows it.
* Measurements in `MEASUREMENTS.md` come from real runs only, with the command, the
  machine and the model digests recorded next to them.
