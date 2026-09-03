"""Audit GroundLM raw result files without rerunning model inference.

The audit uses the corrected working metric implementation and adds the
reviewer-requested parse rates, answer-space profile, per-task Cohen's kappa,
conditional faithfulness, and candidate shortest-path relabeling failures.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable

from groundlm_serialization import (
    TASKS,
    bootstrap_ci,
    compute_distractor_robustness,
    compute_format_stability,
    compute_ground_truth,
    compute_isomorphism_consistency,
    compute_order_sensitivity,
    extract_answer,
    generate_graphs,
)


DIAGNOSTICS = (
    "format_stability",
    "order_sensitivity",
    "isomorphism_consistency",
    "distractor_robustness",
)
RESULT_RE = re.compile(
    r"^results_(?P<model>.+)_(?P<diagnostic>"
    + "|".join(DIAGNOSTICS)
    + r")\.json$"
)


def load_result_sets(results_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load result files, grouped by filename-derived model and diagnostic."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for path in sorted(results_dir.glob("results_*_*.json")):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list in {path}")
        grouped[match.group("model")][match.group("diagnostic")] = value
    return dict(grouped)


def answer_space_profile() -> dict[str, Any]:
    """Reconstruct deterministic dataset answer counts from the base generator."""
    counts: dict[str, Counter[str]] = {task: Counter() for task in TASKS}
    query_count = 0
    for graph_info in generate_graphs():
        for task in TASKS:
            queries = compute_ground_truth(
                graph_info["graph"], task, seed=graph_info["seed"]
            )
            for query in queries:
                counts[task][str(query["answer"])] += 1
                query_count += 1
    return {
        "graphs": 100,
        "graph_task_pairs": 100 * len(TASKS),
        "query_instances": query_count,
        "ground_truth_counts": {
            task: dict(sorted(task_counts.items()))
            for task, task_counts in counts.items()
        },
        "intended_prompts_per_model": {
            "format_stability": query_count * 4,
            "order_sensitivity": query_count * 3,
            "isomorphism_consistency": query_count * 3,
            "distractor_robustness": query_count * 4,
            "total": query_count * 14,
        },
    }


def parse_rates(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return extraction success rates by serialization format and task."""
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    task_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        parsed = extract_answer(result.get("model_answer", ""), result["task"])
        status = "unclear" if parsed == "unclear" else "parsed"
        buckets[result.get("format", "unknown")][status] += 1
        task_buckets[result["task"]][status] += 1

    def summarize(counter: Counter[str]) -> dict[str, Any]:
        total = counter["parsed"] + counter["unclear"]
        return {
            "parsed": counter["parsed"],
            "unclear": counter["unclear"],
            "total": total,
            "parse_success_rate": counter["parsed"] / total if total else None,
        }

    return {
        "by_format": {key: summarize(value) for key, value in sorted(buckets.items())},
        "by_task": {
            key: summarize(value) for key, value in sorted(task_buckets.items())
        },
    }


def canonical_accuracy(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy from canonical edge-list format-stability records."""
    correct = 0
    total = 0
    unclear = 0
    per_task: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        if result.get("format") != "edge_list":
            continue
        if result.get("ordering", "canonical") != "canonical":
            continue
        answer = extract_answer(result.get("model_answer", ""), result["task"])
        is_correct = answer == str(result["ground_truth"])
        correct += int(is_correct)
        total += 1
        unclear += int(answer == "unclear")
        per_task[result["task"]]["correct"] += int(is_correct)
        per_task[result["task"]]["total"] += 1
        per_task[result["task"]]["unclear"] += int(answer == "unclear")

    return {
        "accuracy": correct / total if total else None,
        "correct": correct,
        "total": total,
        "unclear": unclear,
        "per_task": {
            task: {
                "accuracy": values["correct"] / values["total"]
                if values["total"]
                else None,
                **dict(values),
            }
            for task, values in sorted(per_task.items())
        },
    }


def cohen_kappa(left: list[str], right: list[str]) -> dict[str, Any]:
    """Compute nominal Cohen's kappa; `unclear` pairs must be filtered upstream."""
    if len(left) != len(right):
        raise ValueError("Kappa inputs must have the same length")
    n = len(left)
    if n == 0:
        return {"kappa": None, "observed_agreement": None, "expected_agreement": None, "n": 0}

    labels = set(left) | set(right)
    left_counts = Counter(left)
    right_counts = Counter(right)
    observed = sum(a == b for a, b in zip(left, right)) / n
    expected = sum(
        (left_counts[label] / n) * (right_counts[label] / n) for label in labels
    )
    kappa = None if abs(1.0 - expected) < 1e-12 else (observed - expected) / (1.0 - expected)
    return {
        "kappa": kappa,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "n": n,
    }


def ic_kappa(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute original-vs-relabel kappa by task and a macro mean across tasks."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        key = result.get("base_key", result.get("group_key", ""))
        groups[key].append(result)

    paired: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    perturbations = ("neutral_relabel", "misleading_relabel")
    for group in groups.values():
        original_record = next(
            (r for r in group if r.get("perturbation") == "original"), None
        )
        if original_record is None:
            continue
        task = original_record["task"]
        original = extract_answer(original_record.get("model_answer", ""), task)
        for perturbation in perturbations:
            transformed_record = next(
                (r for r in group if r.get("perturbation") == perturbation), None
            )
            if transformed_record is None:
                continue
            transformed = extract_answer(
                transformed_record.get("model_answer", ""), task
            )
            if "unclear" in (original, transformed):
                continue
            paired[perturbation][task].append((original, transformed))

    output: dict[str, Any] = {}
    for perturbation in perturbations:
        per_task: dict[str, Any] = {}
        kappas: list[float] = []
        for task in TASKS:
            pairs = paired[perturbation].get(task, [])
            stat = cohen_kappa(
                [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            )
            per_task[task] = stat
            if stat["kappa"] is not None:
                kappas.append(stat["kappa"])
        output[perturbation] = {
            "macro_kappa_over_defined_tasks": sum(kappas) / len(kappas)
            if kappas
            else None,
            "per_task": per_task,
        }
    return output


def conditional_faithfulness_m(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute P(misleading correct | original correct), counting unclear as wrong."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        key = result.get("base_key", result.get("group_key", ""))
        groups[key].append(result)

    retained: list[float] = []
    for group in groups.values():
        original_record = next(
            (r for r in group if r.get("perturbation") == "original"), None
        )
        transformed_record = next(
            (r for r in group if r.get("perturbation") == "misleading_relabel"),
            None,
        )
        if original_record is None or transformed_record is None:
            continue
        task = original_record["task"]
        ground_truth = str(original_record["ground_truth"])
        original = extract_answer(original_record.get("model_answer", ""), task)
        if original != ground_truth:
            continue
        transformed = extract_answer(
            transformed_record.get("model_answer", ""), task
        )
        retained.append(float(transformed == ground_truth))

    if not retained:
        return {"CF_M": None, "ci_95": None, "retained": 0, "eligible": 0}
    mean, low, high = bootstrap_ci(retained)
    return {
        "CF_M": mean,
        "ci_95": [low, high],
        "retained": int(sum(retained)),
        "eligible": len(retained),
    }


def candidate_hub_failures(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """List distance-3 correct-to-wrong relabeling cases mentioning a Hub mapping."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        key = result.get("base_key", result.get("group_key", ""))
        groups[key].append(result)

    candidates: list[dict[str, Any]] = []
    for key, group in groups.items():
        original_record = next(
            (r for r in group if r.get("perturbation") == "original"), None
        )
        transformed_record = next(
            (r for r in group if r.get("perturbation") == "misleading_relabel"),
            None,
        )
        if original_record is None or transformed_record is None:
            continue
        if original_record.get("task") != "shortest_path":
            continue
        if str(original_record.get("ground_truth")) != "3":
            continue
        mapping = transformed_record.get("relabel_mapping", {})
        if "Hub" not in set(mapping.values()):
            continue
        original = extract_answer(original_record.get("model_answer", ""), "shortest_path")
        transformed = extract_answer(
            transformed_record.get("model_answer", ""), "shortest_path"
        )
        if original == "3" and transformed != "3":
            candidates.append(
                {
                    "base_key": key,
                    "graph_id": original_record.get("graph_id"),
                    "query_original": original_record.get("query"),
                    "query_transformed": transformed_record.get("query"),
                    "answer_original": original,
                    "answer_transformed": transformed,
                    "hub_source_node": next(
                        (source for source, target in mapping.items() if target == "Hub"),
                        None,
                    ),
                    "mapping": mapping,
                }
            )
    return candidates


def audit_model(result_sets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compute all available audits for one model."""
    output: dict[str, Any] = {
        "available_diagnostics": sorted(result_sets),
        "record_counts": {key: len(value) for key, value in sorted(result_sets.items())},
    }
    fs_results = result_sets.get("format_stability")
    if fs_results is not None:
        output["accuracy"] = canonical_accuracy(fs_results)
        output["format_parse_rates"] = parse_rates(fs_results)
        output["FS"] = compute_format_stability(fs_results)

    os_results = result_sets.get("order_sensitivity")
    if os_results is not None:
        output["OS"] = compute_order_sensitivity(os_results)

    ic_results = result_sets.get("isomorphism_consistency")
    if ic_results is not None:
        output["IC"] = compute_isomorphism_consistency(ic_results)
        output["IC_kappa"] = ic_kappa(ic_results)
        output["conditional_faithfulness"] = conditional_faithfulness_m(ic_results)
        output["candidate_distance3_hub_failures"] = candidate_hub_failures(ic_results)

    dr_results = result_sets.get("distractor_robustness")
    if dr_results is not None:
        output["DR"] = compute_distractor_robustness(dr_results)
    return output


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy scalar-like values to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        parser.error(f"Results directory does not exist: {args.results_dir}")
    result_sets = load_result_sets(args.results_dir)
    if not result_sets:
        parser.error(
            "No files matching results_<model>_<diagnostic>.json were found"
        )

    report = {
        "dataset": answer_space_profile(),
        "models": {
            model: audit_model(model_results)
            for model, model_results in sorted(result_sets.items())
        },
    }
    rendered = json.dumps(json_safe(report), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
