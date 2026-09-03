# Evaluation code

## Entry points

- `groundlm_serialization.py`: graph/query generation, serialization,
  perturbations, answer extraction, and diagnostic metrics.
- `run_api_models.py`: asynchronous OpenAI-compatible API runner.
- `run_qwen_local.py`: historical local Qwen3-8B runner.
- `audit_results.py`: offline audit of recovered `results_*.json` files.
- `tests/test_protocol.py`: deterministic protocol and regression checks.
- `legacy/groundlm_serialization_reported.py`: exact GroundLM file from source
  commit `dae0ecd26b5d10a6a3ce5a3f9c4fb600f9f0e2c9`.

The top-level implementation is the later audited working snapshot. It excludes
unparseable comparisons from diagnostic denominators, reports exclusion rates,
removes the canonical self-comparison from corrected OS computation, and
verifies distractors against each exact sampled query. These changes mean it
must not be silently presented as the exact table-producing implementation.
Consult `../docs/PROVENANCE.md` before comparing new output with the paper.

Inference entry points require
`--acknowledge-new-run-not-paper-reproduction`. Direct inference through
`groundlm_serialization.py` is disabled; its `generate` mode remains available.
The API runner refuses to resume into files whose stored model or endpoint does
not match the requested configuration.

The packaged API runner also uses the HTTP client's standard TLS certificate
verification. The recovered runner disabled TLS verification; that unsafe
transport setting is not preserved in the public release copy.
