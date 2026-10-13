# 3. Gate on a significant regression, not on a delta

Status: accepted · 2026-10

## Context

The common CI gate is "fail if the candidate's score is more than X below the baseline".
On a golden set of a few dozen cases that gate is mostly noise: two flipped cases out of
sixty move a pass rate by 3.3 points. A gate that fails on noise gets disabled; a gate
that is loosened until it stops failing on noise also stops catching regressions.

## Decision

* Compare runs **on paired cases** — the same case ids in both runs — because each case
  serves as its own control and only discordant pairs carry information.
* For pass/fail metrics use the **exact McNemar test** on the discordant pairs; for
  continuous scores use a **paired bootstrap** of the mean difference. Report the delta
  with a 95% interval either way.
* **Fail only when both hold:** the drop is at least `max_drop` *and* `p < alpha`.
  * A large drop that is not significant passes with a warning that names the sample size.
  * A significant drop smaller than `max_drop` passes with a warning.
* Refuse to gate (exit 4) when there are fewer than `min_cases` paired cases or the two runs
  used different datasets (the dataset hash differs).
* Budgets — p95 latency and cost — are separate checks with their own exit code (3).

## Consequences

* A real but small regression on a small golden set can pass with a warning. That is the
  honest outcome: the data cannot distinguish it from noise. The fix is more cases, and
  the warning says so.
* Exit codes are distinct (0 pass, 1 regression, 3 budget, 4 cannot gate, 2 usage) so a
  pipeline can treat them differently.
* Testing several metrics at α = 0.05 each inflates the family-wise error rate. The gate
  checks one metric by design; the others are reported, not gated.
