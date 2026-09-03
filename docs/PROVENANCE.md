# Provenance and reproducibility record

## Source snapshots

- Source repository recorded in the available checkout:
  `https://github.com/Anormalm/AAAI.git`
- Source commit: `dae0ecd26b5d10a6a3ce5a3f9c4fb600f9f0e2c9`
- Commit date: 2026-07-02 15:48:07 +08:00
- Commit subject: `Initial commit: GroundLM paper + S3D GNN pipeline code`

`legacy/groundlm_serialization_reported.py` is byte-for-byte the GroundLM file
from that commit. The top-level `groundlm_serialization.py` came from the later,
uncommitted working tree, which differed by 68 insertions and 10 deletions when
this artifact was assembled. The model runners were copied from the same
working tree; neither runner differed from the commit at assembly time.

For this public release tree, `run_gpt4o_mini.py` is named
`code/run_api_models.py`, its usage text reflects that name, and its HTTP clients
use standard TLS certificate verification. The recovered runner's
`verify=False` transport setting is intentionally not preserved.

The packaged working snapshot has one additional, narrowly scoped correction:
distractor verification now computes the symbolic answer for the exact sampled
source nodes. The prior check reran the query sampler after adding distractor
nodes; because that changes the sampling population, it usually could not find
the original query and skipped valid cases.

## Recovered experimental protocol

- Seed: 42.
- Graphs: 100 connected Erdos-Renyi graphs, with 20 graphs at each size in
  `{5, 8, 10, 12, 15}` and edge probability 0.3. Disconnected draws are retried
  and finally connected if necessary.
- Tasks: connectivity, shortest path, triangle participation, and common
  neighbor.
- Queries: three sampled queries per graph and task: 1,200 query instances in
  total, corresponding to 400 graph-task pairs.
- Formats: edge list, adjacency list, JSON-like nodes/edges, and natural
  language.
- Order conditions: canonical, one seeded random ordering, and one BFS ordering.
- Relabel conditions: original, one neutral relabeling, and one semantic-name
  relabeling per instance. All nodes are relabeled automatically.
- Distractor conditions: baseline plus three named distractor types.
- Intended calls per model: 4,800 format, 3,600 order, 3,600 relabeling, and
  4,800 distractor prompts, for 16,800 calls total. Five reported model
  configurations would therefore require 84,000 calls before retries.
- Metrics pool comparison indicators across all instances. They are not
  computed as an unweighted macro-average of task-level metrics. Each task has
  the same number of sampled queries in the base benchmark.
- Format stability is the mean agreement over all six unordered pairs of the
  four formats; it is not agreement against one reference format.
- Order sensitivity compares two noncanonical orderings (random and BFS) to the
  canonical baseline in the corrected working snapshot. The archived snapshot
  also included a canonical self-comparison, which mechanically contributed a
  zero flip.
- Confidence intervals use 1,000 seed-42 percentile bootstrap resamples of the
  individual comparison indicators. This captures item/comparison sampling
  uncertainty, not model-sampling variability or graph-cluster dependence.

## Unresolved provenance issues

These points must be resolved from the original experiment machine before the
artifact or a reviewer answer claims more than the available evidence supports.

1. **Raw generations are absent.** No `results_*.json` or paper-producing
   `metrics_*.json` files were present in the source checkout.
2. **`CF_M` is absent from the source.** The paper reports conditional
   faithfulness, but no available Python file computes it. `audit_results.py`
   implements the definition stated in the camera-ready source:
   `Pr[prediction_misleading = y | prediction_original = y]`; its output must be
   checked against the recovered raw generations.
3. **The semantic mapping conflicts with its description.** The available code
   sorts nodes by decreasing degree and assigns `Hub`, `Center`, `Bridge`, ...
   in that order. Consequently, `Hub` is assigned to a maximum-degree node in
   every generated base graph. This is role-congruent by degree and cannot
   generate the paper's example in which a peripheral node is renamed `Hub`.
   The exact relabeling implementation used for the reported table and failure
   example must be recovered.
4. **Two distractor names are identical implementations.** Both
   `disconnected_component` and `disconnected_triangle` add a disconnected
   three-node triangle. The aggregate DR calculation therefore weights that
   transformation twice. This has been preserved for provenance rather than
   silently redefining the reported diagnostic.
5. **Unparseable-answer policy changed.** The archived code does not implement
   the paper's stated policy consistently. The working snapshot treats
   `unclear` as missing and reports exclusion rates. Raw outputs are necessary
   to report both include-as-disagreement and exclude-with-parse-rate variants.
6. **Reasoning configuration is only partly established.** The runner explicitly
   disables reasoning for Qwen and DeepSeek OpenRouter calls. It does not pass
   an explicit reasoning setting for gpt-5.4; it only uses temperature 0 and a
   256-token completion limit. Therefore this source cannot substantiate the
   claim that reasoning was disabled for gpt-5.4.
7. **No arbitrary-identifier or role-congruent ablation is present.** The prompt
   never tells the model that identifiers are arbitrary, and there is no
   separately defined congruent condition.

## Recovery checklist

- Copy all paper-producing `results_*.json`, prompt manifests, metrics files,
  logs, and environment/package manifests from the experiment machine.
- Record the exact model IDs, providers, dates, endpoint settings, and commands.
- Recover the relabeling code and the exact shortest-path `Hub` example.
- Run `python audit_results.py --results-dir <dir> --output audit_summary.json`.
- Compare the audited metrics and denominators with every table entry.
- Decide whether to publish raw generations in Git or as a versioned release
  asset, after removing private provider metadata.
- Add a software license and citation metadata.

## Public-release interpretation

The prioritized claim boundary is maintained in `../RELEASE_STATUS.md`; the
machine-readable status is in `../protocol_manifest.json`. New runner outputs
carry a `run_manifest.json` and an explicit non-reproduction label. These
release safeguards do not resolve any historical provenance gap.
