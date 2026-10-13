# 2. Local-first providers over plain HTTP

Status: accepted · 2026-10

## Context

An evaluation that only runs with an API key and a budget gets run less often, and its
results cannot be reproduced by anyone without the same account. Hosted judges also change
under you: a model alias can point at new weights between two runs.

## Decision

* **Ollama is the default provider.** A local model with `temperature: 0` and a fixed
  `seed` gives repeatable verdicts at zero marginal cost, and the run records the Ollama
  version and the model digest so the result can be tied to exact weights.
* **Anthropic and OpenAI are optional**, reached over plain HTTP with `httpx` and a key
  from the environment. No vendor SDK is a dependency.
* **Every call goes through a content-addressed cache** keyed on provider, model,
  endpoint, messages and generation parameters. An unchanged suite re-runs from cache;
  changing one prompt re-runs only the calls it affects.
* Reasoning models run with thinking off by default (`think: false`) for speed and
  repeatability; a suite can turn it on.

## Consequences

* A local 8B judge is weaker than a frontier model. That is acceptable precisely because
  ADR 1 measures and corrects the judge rather than trusting it — a weaker, calibrated
  judge beats a stronger, uncalibrated one for gating.
* Determinism is only as good as the backend's seeding. The measurements re-run a sample
  without the cache to check it.
* Cost reporting needs a price table in the suite for hosted models; local runs report $0.
