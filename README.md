# Graph Evidence Is Not Text Evidence

Evaluation code for **Graph Evidence Is Not Text Evidence: Serialization
Faithfulness as Structural Invariance in Graph-Grounded Language Models**,
accepted at the EMNLP GroundLM 2026 Workshop.

This is a code-only repository. The camera-ready paper and poster are not
versioned here.

## Repository layout

```text
code/        Prompt generation, inference runners, metrics, auditor, and tests
docs/        Technical provenance and reproducibility notes
results/     Instructions for adding recovered raw model generations
scripts/     PowerShell test entry point
```

## Read this before reproducing the table

The source checkout used to assemble this repository did not contain the raw
model generations behind the paper table. It also did not contain the original
implementation of the reported conditional-faithfulness metric `CF_M`.
Furthermore, the available semantic-name relabeler assigns names after sorting
nodes by decreasing degree, so `Hub` is assigned to a maximum-degree node. The
release therefore describes this condition as **semantic-name relabeling**, not
as a verified role-incongruent intervention.

This repository does not claim bit-for-bit numerical reproduction until the
original results are recovered and audited. See
[RELEASE_STATUS.md](RELEASE_STATUS.md) for the claim boundary and prioritized
risk register, and [docs/PROVENANCE.md](docs/PROVENANCE.md) for the full record.

## Run the code checks

Python 3.10 or newer is recommended.

```bash
cd code
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements.txt` gives supported ranges. `requirements-lock.txt` records the
exact clean-checkout verification environment for this release; it is not the
unrecovered historical experiment environment.

On Windows, the same check can be run from the repository root with:

```powershell
.\scripts\test.ps1
```

## Generate prompts or run a model

The API runner supports the paper model identifiers through OpenAI-compatible
endpoints:

```bash
cd code
python run_api_models.py \
  --model deepseek/deepseek-v4-flash \
  --endpoint openrouter \
  --acknowledge-new-run-not-paper-reproduction \
  --output-dir ../results
```

Set provider keys only through environment variables. Never commit `.env`
files or credentials. The local `run_qwen_local.py` script is a historical
Qwen3-8B runner; it is not the Qwen3.6-35B-A3B configuration reported in the
paper.

When original results are available, audit them without rerunning inference:

```bash
cd code
python audit_results.py --results-dir ../results \
  --output ../results/audit_summary.json
```

The auditor reports diagnostic point estimates and intervals, per-format parse
rates, answer-space distributions, per-task Cohen's kappa, a reconstructed
`CF_M`, and candidate distance-3 semantic-relabeling failures. Its JSON output
is explicitly marked as a reconstructed audit rather than the paper-producing
analysis.

## License

Code is released under the [MIT License](LICENSE). Paper text, model outputs,
and third-party materials are not relicensed by this repository.
