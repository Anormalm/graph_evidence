"""
Run OpenAI model experiments for Graph Serialization Faithfulness.

Usage:
  # Via standard OpenAI API:
  export OPENAI_API_KEY='sk-...'
  python3 run_api_models.py --model gpt-5.4-mini-2026-03-17
  python3 run_api_models.py --model gpt-5.4-2026-03-05

  # Via ByteDance proxy (GPT-5.5):
  export BYTEDANCE_API_KEY='your_ak'
  python3 run_api_models.py --model gpt-5.5-2026-04-24 --endpoint bytedance

This will:
1. Generate all prompts (if not already generated)
2. Run inference for all 4 RQs (async, ~20 concurrent)
3. Compute metrics and print the main result table
4. Save all results to ./groundlm_output_v2/
"""

import os
import json
import time
import sys
import asyncio
import argparse
from datetime import datetime, timezone
import platform
import subprocess

# Add parent dir if needed
sys.path.insert(0, os.path.dirname(__file__))

from groundlm_serialization import (
    generate_all_prompts, OUTPUT_DIR,
    compute_format_stability, compute_order_sensitivity,
    compute_isomorphism_consistency, compute_distractor_robustness,
    extract_answer, TASKS, FORMATS, ORDERINGS
)

# Map full model IDs to short names for file naming
SHORT_NAMES = {
    "gpt-5.4-mini-2026-03-17": "gpt-5.4-mini",
    "gpt-5.4-2026-03-05": "gpt-5.4",
    "gpt-5.5-2026-04-24": "gpt-5.5",
    "qwen/qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
    "deepseek/deepseek-v4-flash": "DeepSeek-V4-Flash",
    "meta-llama/llama-4-scout": "Llama-4-Scout",
}

# ByteDance proxy config
BYTEDANCE_BASE_URL = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl/openai/deployments/gpt_openapi"

# OpenRouter config
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_client(model, endpoint="openai"):
    """Create an AsyncOpenAI client for the given endpoint."""
    import httpx
    from openai import AsyncOpenAI

    if endpoint == "bytedance":
        ak = os.environ.get("BYTEDANCE_API_KEY", "")
        if not ak:
            raise ValueError("BYTEDANCE_API_KEY not set. Run: export BYTEDANCE_API_KEY='your_ak'")
        http_client = httpx.AsyncClient()
        client = AsyncOpenAI(
            base_url=BYTEDANCE_BASE_URL,
            api_key=ak,
            default_headers={"Api-Key": ak},
            http_client=http_client,
        )
        return client
    elif endpoint == "openrouter":
        ak = os.environ.get("OPENROUTER_API_KEY", "")
        if not ak:
            raise ValueError("OPENROUTER_API_KEY not set. Run: export OPENROUTER_API_KEY='your_key'")
        http_client = httpx.AsyncClient()
        client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=ak,
            http_client=http_client,
        )
        return client
    else:
        # Standard OpenAI API
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='sk-...'")
        http_client = httpx.AsyncClient()
        client = AsyncOpenAI(
            api_key=api_key,
            http_client=http_client,
        )
        return client


async def query_openai_async(prompts, model="gpt-5.4-mini-2026-03-17",
                              endpoint="openai", output_file=None, max_concurrent=20):
    """Async OpenAI query with concurrency control."""
    client = make_client(model, endpoint)
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(prompts)

    async def query_one(idx, p):
        async with semaphore:
            max_retries = 5
            for attempt in range(max_retries):
                answer_text = "ERROR: query failed"
                prompt_tokens = 0
                completion_tokens = 0
                confidence = None
                try:
                    api_kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a precise graph reasoning assistant. Answer graph questions based only on the provided graph description. Give your final answer clearly."},
                            {"role": "user", "content": p["prompt"]},
                        ],
                        "temperature": 0.0,
                    }
                    # Use max_completion_tokens for GPT-5.x models
                    if model.startswith(("gpt-5", "o3", "o4")):
                        api_kwargs["max_completion_tokens"] = 256
                    elif "qwen" in model.lower() or "qwen/qwen" in model.lower():
                        api_kwargs["max_tokens"] = 256
                        # Disable reasoning/thinking mode for Qwen models
                        api_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
                    elif "deepseek" in model.lower():
                        api_kwargs["max_tokens"] = 256
                        # DeepSeek V4: disable reasoning for non-thinking evaluation
                        api_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
                    elif "llama" in model.lower():
                        api_kwargs["max_tokens"] = 256
                        # Llama 4 Scout: no special thinking mode, standard params
                    else:
                        api_kwargs["max_tokens"] = 256
                        api_kwargs["logprobs"] = True
                        api_kwargs["top_logprobs"] = 5

                    # ByteDance proxy requires extra header
                    if endpoint == "bytedance":
                        api_kwargs["extra_headers"] = {"X-TT-LOGID": ""}

                    response = await client.chat.completions.create(**api_kwargs)
                    answer_text = response.choices[0].message.content.strip()
                    usage = response.usage
                    prompt_tokens = usage.prompt_tokens if usage else 0
                    completion_tokens = usage.completion_tokens if usage else 0
                    try:
                        if response.choices[0].logprobs and response.choices[0].logprobs.content:
                            first_token_lp = response.choices[0].logprobs.content[0]
                            import math
                            confidence = math.exp(first_token_lp.logprob)
                    except Exception:
                        pass
                    break  # success — exit retry loop
                except Exception as e:
                    err_str = str(e)
                    is_retryable = any(k in err_str.lower() for k in
                                       ["429", "rate", "limit", "overload", "timeout", "503", "connection"])
                    if is_retryable and attempt < max_retries - 1:
                        wait = min(2 ** attempt + 0.5 * attempt, 30)  # exp backoff, cap 30s
                        await asyncio.sleep(wait)
                        continue
                    answer_text = f"ERROR: {err_str}"
                    prompt_tokens = 0
                    completion_tokens = 0

            result = dict(p)
            result["model_answer"] = answer_text
            result["model"] = model
            result["endpoint"] = endpoint
            result["prompt_tokens"] = prompt_tokens
            result["completion_tokens"] = completion_tokens
            result["confidence"] = confidence
            results[idx] = result

    print(f"  Launching {len(prompts)} queries via {endpoint} (max {max_concurrent} concurrent)...", flush=True)
    t0 = time.time()

    async def track_progress():
        """Background task that prints progress every 30s and saves partial results."""
        while True:
            await asyncio.sleep(30)
            done = sum(1 for r in results if r is not None)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(prompts) - done) / rate if rate > 0 else 0
            print(f"    {done}/{len(prompts)} done ({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)
            if output_file and done > 0:
                completed = [r for r in results if r is not None]
                with open(output_file, "w") as f:
                    json.dump(completed, f, indent=2)

    tracker = asyncio.create_task(track_progress())
    tasks = [query_one(i, p) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)
    tracker.cancel()

    elapsed = time.time() - t0
    rate = len(prompts) / elapsed if elapsed > 0 else 0
    print(f"    {len(prompts)}/{len(prompts)} done ({rate:.1f}/s, total {elapsed/60:.1f} min)", flush=True)

    results = [r for r in results if r is not None]

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {output_file}")

    return results


async def run_all(model="gpt-5.4-mini-2026-03-17", endpoint="openai",
                  use_expanded=False, output_dir="./groundlm_output_v2"):
    os.makedirs(output_dir, exist_ok=True)
    short = SHORT_NAMES.get(model, model)

    # Step 1: Generate prompts
    print("=" * 60)
    print("Step 1: Generating prompts...")
    print("=" * 60)
    all_prompts = generate_all_prompts(output_dir, use_expanded=use_expanded)

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    run_manifest = {
        "artifact_claim": "new_run_of_released_protocol_not_paper_table_reproduction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "python": platform.python_version(),
        "requested_model_id": model,
        "endpoint": endpoint,
        "temperature": 0.0,
        "maximum_completion_tokens": 256,
        "expanded_graphs": use_expanded,
        "semantic_relabeling": "degree_congruent_semantic_names_unverified_against_paper",
        "reasoning_setting": "explicitly_disabled_for_qwen_and_deepseek_only; otherwise_unspecified",
        "prompt_counts": {name: len(prompts) for name, prompts in all_prompts.items()},
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, sort_keys=True)

    # Step 2: Run inference for each RQ
    print("\n" + "=" * 60)
    print(f"Step 2: Running {model} via {endpoint}...")
    print("=" * 60)

    all_results = {}

    for rq_name in ["format_stability", "order_sensitivity", "isomorphism_consistency", "distractor_robustness"]:
        prompts = all_prompts[rq_name]
        results_file = os.path.join(output_dir, f"results_{short}_{rq_name}.json")

        if os.path.exists(results_file):
            with open(results_file) as f:
                existing = json.load(f)
            incompatible = [
                record for record in existing
                if record.get("model") != model or record.get("endpoint") != endpoint
            ]
            if incompatible:
                raise RuntimeError(
                    f"Refusing to mix {len(incompatible)} existing records with "
                    f"requested model={model!r}, endpoint={endpoint!r}: {results_file}"
                )
            if len(existing) == len(prompts):
                print(f"\n  {rq_name}: Loading existing results ({len(existing)}) from {results_file}")
                results = existing
            else:
                print(f"\n  {rq_name}: Partial results ({len(existing)}/{len(prompts)}), resuming {len(prompts) - len(existing)} missing...")
                # Build set of already-completed prompt indices by matching prompt text
                done_prompts = {r.get("prompt", "") for r in existing}
                remaining = [p for p in prompts if p.get("prompt", "") not in done_prompts]
                if remaining:
                    t0 = time.time()
                    new_results = await query_openai_async(remaining, model=model, endpoint=endpoint,
                                                        output_file=None, max_concurrent=30 if endpoint == "openrouter" else 20)
                    elapsed = time.time() - t0
                    print(f"  Done in {elapsed:.1f}s ({len(new_results)} new results)")
                    results = existing + new_results
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    results = existing
        else:
            print(f"\n  {rq_name}: Querying {len(prompts)} prompts...")
            t0 = time.time()
            results = await query_openai_async(prompts, model=model, endpoint=endpoint,
                                                output_file=results_file, max_concurrent=30 if endpoint == "openrouter" else 20)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s ({len(results)} results)")

        all_results[rq_name] = results

    # Step 3: Compute accuracy
    print("\n" + "=" * 60)
    print("Step 3: Computing task accuracy...")
    print("=" * 60)

    correct = 0
    total = 0
    for r in all_results["format_stability"]:
        if r["format"] == "edge_list" and r["ordering"] == "canonical":
            extracted = extract_answer(r["model_answer"], r["task"])
            gt = r["ground_truth"]
            if extracted == gt:
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    print(f"  Task accuracy (edge_list, canonical): {accuracy:.4f} ({correct}/{total})")

    # Step 4: Compute faithfulness metrics
    print("\n" + "=" * 60)
    print("Step 4: Computing diagnostics for this new run...")
    print("=" * 60)

    metrics = {}
    metrics["FS"] = compute_format_stability(all_results["format_stability"])
    metrics["OS"] = compute_order_sensitivity(all_results["order_sensitivity"])
    metrics["IC"] = compute_isomorphism_consistency(all_results["isomorphism_consistency"])
    metrics["DR"] = compute_distractor_robustness(all_results["distractor_robustness"])

    # Step 5: Print results
    print("\n" + "=" * 80)
    print(f"NEW-RUN DIAGNOSTICS (NOT PAPER REPRODUCTION) — {model}")
    print("=" * 80)

    fs = metrics["FS"]["FS"]
    fs_ci = metrics["FS"].get("FS_ci", (0.0, 0.0))
    os_m = metrics["OS"]["OS"]
    os_ci = metrics["OS"].get("OS_ci", (0.0, 0.0))
    ic_n = metrics["IC"]["IC_neutral"]
    ic_n_ci = metrics["IC"].get("IC_neutral_ci", (0.0, 0.0))
    ic_m = metrics["IC"]["IC_misleading"]
    ic_m_ci = metrics["IC"].get("IC_misleading_ci", (0.0, 0.0))
    dr = metrics["DR"]["DR"]
    dr_ci = metrics["DR"].get("DR_ci", (0.0, 0.0))

    print(f"  {'Model':<25} {'Accuracy':>10} {'FS':>14} {'OS':>14} {'IC-N':>14} {'IC-M':>14} {'DR':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {short:<25} {accuracy:>10.4f} {fs:>7.4f}({fs_ci[0]:.2f}-{fs_ci[1]:.2f}) {os_m:>7.4f}({os_ci[0]:.2f}-{os_ci[1]:.2f}) {ic_n:>7.4f}({ic_n_ci[0]:.2f}-{ic_n_ci[1]:.2f}) {ic_m:>7.4f}({ic_m_ci[0]:.2f}-{ic_m_ci[1]:.2f}) {dr:>7.4f}({dr_ci[0]:.2f}-{dr_ci[1]:.2f})")

    # Per-task breakdown
    print("\n--- Per-Task Breakdown ---")
    print(f"  {'Task':<20} {'Acc':>8} {'FS':>8} {'OS':>8} {'IC-N':>8} {'IC-M':>8} {'DR':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for task in TASKS:
        t_correct = 0
        t_total = 0
        for r in all_results["format_stability"]:
            if r["format"] == "edge_list" and r["ordering"] == "canonical" and r["task"] == task:
                extracted = extract_answer(r["model_answer"], task)
                if extracted == r["ground_truth"]:
                    t_correct += 1
                t_total += 1
        t_acc = t_correct / max(t_total, 1)

        t_fs = metrics["FS"]["FS_per_task"].get(task, 0.0)
        t_os = metrics["OS"]["OS_per_task"].get(task, 0.0)
        t_ic_n = metrics["IC"]["IC_neutral_per_task"].get(task, 0.0)
        t_ic_m = metrics["IC"]["IC_misleading_per_task"].get(task, 0.0)
        t_dr = metrics["DR"]["DR_per_task"].get(task, 0.0)
        print(f"  {task:<20} {t_acc:>8.4f} {t_fs:>8.4f} {t_os:>8.4f} {t_ic_n:>8.4f} {t_ic_m:>8.4f} {t_dr:>8.4f}")

    # Per-format stability pairs
    print("\n--- Format Stability by Format Pair ---")
    for pair_key, val in sorted(metrics["FS"]["FS_per_pair"].items()):
        print(f"  {pair_key}: {val:.4f}")

    # Per-ordering sensitivity
    print("\n--- Order Sensitivity by Ordering ---")
    for ordering, val in sorted(metrics["OS"]["OS_per_ordering"].items()):
        print(f"  {ordering}: {val:.4f}")

    # Per-distractor robustness
    print("\n--- Distractor Robustness by Distractor Type ---")
    for dtype, val in sorted(metrics["DR"]["DR_per_distractor"].items()):
        print(f"  {dtype}: {val:.4f}")

    # Save metrics
    summary = {
        "artifact_claim": "new_run_of_released_protocol_not_paper_table_reproduction",
        "model": short,
        "model_id": model,
        "endpoint": endpoint,
        "accuracy": accuracy,
        "FS": fs,
        "OS": os_m,
        "IC_neutral": ic_n,
        "IC_misleading": ic_m,
        "DR": dr,
        "FS_per_task": metrics["FS"]["FS_per_task"],
        "OS_per_task": metrics["OS"]["OS_per_task"],
        "IC_neutral_per_task": metrics["IC"]["IC_neutral_per_task"],
        "IC_misleading_per_task": metrics["IC"]["IC_misleading_per_task"],
        "DR_per_task": metrics["DR"]["DR_per_task"],
        "FS_per_pair": metrics["FS"]["FS_per_pair"],
        "OS_per_ordering": metrics["OS"]["OS_per_ordering"],
        "DR_per_distractor": metrics["DR"]["DR_per_distractor"],
    }

    summary_path = os.path.join(output_dir, f"metrics_{short}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5.4-2026-03-05",
                        help="Model ID (e.g. gpt-5.4-mini-2026-03-17, qwen/qwen3.6-35b-a3b)")
    parser.add_argument("--endpoint", type=str, default="openai", choices=["openai", "bytedance", "openrouter"],
                        help="API endpoint: 'openai', 'bytedance', or 'openrouter'")
    parser.add_argument("--expanded", action="store_true", help="Use expanded graph types")
    parser.add_argument("--output-dir", type=str, default="./groundlm_output_v2", help="Output directory")
    parser.add_argument(
        "--acknowledge-new-run-not-paper-reproduction",
        action="store_true",
        help="Required acknowledgement that this run cannot reproduce the paper table",
    )
    args = parser.parse_args()

    if not args.acknowledge_new_run_not_paper_reproduction:
        parser.error(
            "This released snapshot is not the verified paper-producing protocol. "
            "Read RELEASE_STATUS.md, then pass "
            "--acknowledge-new-run-not-paper-reproduction to launch a new run."
        )

    # Check API key based on endpoint
    if args.endpoint == "bytedance":
        if "BYTEDANCE_API_KEY" not in os.environ:
            print("ERROR: BYTEDANCE_API_KEY not set.")
            print("Run: export BYTEDANCE_API_KEY='your_ak'")
            sys.exit(1)
    elif args.endpoint == "openrouter":
        if "OPENROUTER_API_KEY" not in os.environ:
            print("ERROR: OPENROUTER_API_KEY not set.")
            print("Run: export OPENROUTER_API_KEY='your_key'")
            sys.exit(1)
    else:
        if "OPENAI_API_KEY" not in os.environ:
            print("ERROR: OPENAI_API_KEY not set.")
            print("Run: export OPENAI_API_KEY='sk-...'")
            sys.exit(1)

    asyncio.run(run_all(model=args.model, endpoint=args.endpoint,
                        use_expanded=args.expanded, output_dir=args.output_dir))
