# Results directory

The raw paper-producing model generations were not present in the recovered
source checkout and are intentionally not fabricated here.

Place recovered files here using the naming convention:

```text
results_<model>_format_stability.json
results_<model>_order_sensitivity.json
results_<model>_isomorphism_consistency.json
results_<model>_distractor_robustness.json
```

Then run:

```bash
cd ../code
python audit_results.py --results-dir ../results \
  --output ../results/audit_summary.json
```

Raw generations are ignored by Git by default because they can be large and
may contain provider metadata. Compact audit summaries are not ignored.
