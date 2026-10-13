# Offline example: a ticket router

No model, no network — this is what CI runs on every commit to prove the gate does what
it says. The system under test is a keyword router that sends support tickets to one of
five queues; the golden set is 60 tickets with the queue a person would choose.

| Candidate | What changed | Paired outcome vs baseline | Gate |
|---|---|---|---|
| `regressed_route` | billing keywords "tidied" | 11 tickets worse, 0 better | **fail, exit 1** — Δ −18.3 pts, McNemar p = 0.001 |
| `noisy_route` | one keyword added | 2 worse, 0 better | **pass with a warning** — Δ −3.3 pts is beyond the 2-point tolerance, but p = 0.5 |
| `refactored_route` | same rules, restructured | identical | **pass** |

The middle row is the point of the project. A gate that compares two pass rates would fail
that change: 51/60 is below 53/60 by more than the tolerance. Two flipped tickets out of
sixty is not evidence of anything, and evalproof says so instead of blocking a merge.

```bash
for v in baseline regressed noisy refactored; do
  uv run evalproof run examples/deterministic/suite.yaml \
    --system python:router:${v}_route --out examples/deterministic/runs/$v.json --quiet
done
uv run evalproof compare examples/deterministic/runs/baseline.json \
  examples/deterministic/runs/regressed.json --suite examples/deterministic/suite.yaml --gate
echo $?   # 1
```

The four run files in `runs/` are the committed output of exactly those commands.
