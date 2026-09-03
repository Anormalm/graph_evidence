# Release status and claim boundaries

This repository is a **partial research artifact**, not a verified reproduction
package for the paper's numerical table. It contains a preserved source snapshot,
an audited later snapshot, inference runners, and diagnostic tooling. The raw
paper-producing generations, the original `CF_M` implementation, and the exact
role-incongruent relabeling implementation have not yet been recovered.

## What can be claimed now

- The archived file in `code/legacy/` is byte-identical to the file in source
  commit `dae0ecd26b5d10a6a3ce5a3f9c4fb600f9f0e2c9`.
- The deterministic graph/query workload and the behavior tested by
  `code/tests/test_protocol.py` can be regenerated.
- New experiments can be run with the released snapshot if they are described
  as **new runs of the released protocol**, not reproductions of the paper table.
- Recovered raw files can be inspected with a reconstructed auditor, whose
  outputs must be described as audit diagnostics until reconciled with the
  original table-producing analysis.

## What must not be claimed yet

- That the paper's numerical table is reproducible from this repository.
- That `CF_M` in `audit_results.py` is the original table-producing metric code.
- That the released semantic-name relabeler implements a role-incongruent or
  deliberately misleading intervention. It assigns `Hub` to a maximum-degree
  node and is role-congruent by degree.
- That the two identically implemented triangle distractors are independent
  perturbations, or that their pooled aggregate is an unweighted summary of
  three distinct transformations.
- That reasoning was disabled for every reported provider/model configuration.

## Risk register

| Priority | Gap | Threatened claim | Required resolution |
|---|---|---|---|
| P0 | Raw generations and table-producing metrics absent | All numerical reproduction and denominator checks | Recover immutable raw outputs, manifests, logs, and metrics; publish checksums; reconcile every table cell |
| P0 | Released semantic mapping conflicts with the paper intervention | Role-incongruent relabeling and causal interpretation of semantic failures | Recover exact code and cited example; otherwise rename/restate the condition and rerun |
| P0 | Original `CF_M` code absent | Exact conditional-faithfulness values | Recover the implementation or treat the released definition as a new, explicitly reconstructed analysis |
| P1 | Unparseable-output policy differs across snapshots | FS, OS, IC, DR denominators and comparability | Report both policies from raw outputs, including parse/exclusion counts |
| P1 | Two distractor labels implement the same triangle | Aggregate DR weighting and diversity of interventions | Preserve for paper audit; use deduplicated conditions for a clearly versioned new protocol |
| P1 | Model/provider reasoning settings incomplete | Cross-model comparability | Recover request logs/configuration; otherwise mark affected configurations unknown |
| P2 | Item-level bootstrap ignores graph clustering and model sampling | Precision of uncertainty intervals | Add graph-cluster bootstrap and multiple inference seeds to future runs; do not reinterpret old intervals |
| P2 | No arbitrary-identifier or congruent-control ablation | Mechanism attribution to semantic labels | Add controls only as a new experiment, not a reconstruction |

## Two separate workflows

### A. Paper recovery and audit

Do not rerun models as a substitute for missing historical evidence. Recover the
original files, preserve them read-only, record SHA-256 checksums, and run the
auditor. Reconcile discrepancies before changing any paper claim.

### B. New replication using this release

Run the current code only with the explicit acknowledgement flag. Archive the
generated `run_manifest.json`, prompt files, raw outputs, package environment,
and provider-returned model identifiers. Report the semantic condition as
`degree-congruent semantic-name relabeling`.

The detailed chain of custody and recovery checklist are in
[`docs/PROVENANCE.md`](docs/PROVENANCE.md).
