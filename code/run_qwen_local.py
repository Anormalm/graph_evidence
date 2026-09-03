"""
Run Qwen3-8B local inference for Graph Serialization Faithfulness.

Usage:
  python3 run_qwen_local.py                                    # Qwen3-8B on GPU/MPS
  python3 run_qwen_local.py --model-path /path/to/model        # custom path
  python3 run_qwen_local.py --device cpu                       # force CPU
  python3 run_qwen_local.py --no-vllm                          # use transformers
  python3 run_qwen_local.py --output-dir ./groundlm_output_v2  # custom output dir

Uses Qwen3-8B in NON-THINKING mode (no <think> tokens) by default.
Thinking mode results can go in an appendix if desired.

This will:
1. Generate all prompts (if not already generated)
2. Run inference for all 4 RQs using vLLM or transformers
3. Compute metrics and print the main result table
4. Save all results to output directory
"""

import os
import json
import time
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from groundlm_serialization import (
    generate_all_prompts, OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
    compute_format_stability, compute_order_sensitivity,
    compute_isomorphism_consistency, compute_distractor_robustness,
    extract_answer, TASKS, FORMATS, ORDERINGS
)

SYSTEM_PROMPT = "You are a precise graph reasoning assistant. Answer graph questions based only on the provided graph description. Give your final answer clearly."

DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/huggingface/Qwen3-8B")

# Qwen3 non-thinking mode: add /no_think suffix to system prompt
# This tells Qwen3 to skip the <think>...</think> chain-of-thought block
SYSTEM_PROMPT_NOTHINK = SYSTEM_PROMPT + " /no_think"


def run_inference_vllm(prompts, model_path, max_concurrent=8):
    """Run inference using vLLM (fast, recommended)."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  vLLM not installed, falling back to transformers.")
        return None

    print(f"  Loading model with vLLM: {model_path}")
    llm = LLM(model=model_path, trust_remote_code=True, dtype="auto")

    sampling = SamplingParams(temperature=0.0, max_tokens=512)

    # Build full prompts with Qwen3 chat template + no_think
    full_prompts = []
    for p in prompts:
        full_prompts.append(
            f"<|im_start|>system\n{SYSTEM_PROMPT_NOTHINK}<|im_end|>\n"
            f"<|im_start|>user\n{p['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    print(f"  Running {len(full_prompts)} prompts through vLLM...")
    t0 = time.time()
    outputs = llm.generate(full_prompts, sampling)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)")

    results = []
    for i, (p, out) in enumerate(zip(prompts, outputs)):
        answer = out.outputs[0].text.strip()
        result = dict(p)
        result["model_answer"] = answer
        result["model"] = "Qwen3-8B"
        result["prompt_tokens"] = len(out.prompt_token_ids)
        result["completion_tokens"] = len(out.outputs[0].token_ids)
        result["confidence"] = None
        results.append(result)

    return results


def run_inference_transformers(prompts, model_path, device="auto", batch_size=1):
    """Run inference using transformers + accelerate (slower but no vLLM dependency)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model with transformers: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    model.eval()

    results = []
    total = len(prompts)
    t0 = time.time()

    for i, p in enumerate(prompts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NOTHINK},
            {"role": "user", "content": p["prompt"]},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: disable thinking mode
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,  # greedy when do_sample=False
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip any remaining <think>...</think> blocks
        if "<think>" in answer:
            import re
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

        result = dict(p)
        result["model_answer"] = answer
        result["model"] = "Qwen3-8B"
        result["prompt_tokens"] = inputs["input_ids"].shape[1]
        result["completion_tokens"] = len(new_tokens)
        result["confidence"] = None
        results.append(result)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"    {i+1}/{total} done ({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({total/elapsed:.1f} prompts/s)")
    return results


def run_rq(prompts, rq_name, model_path, output_dir, device, use_vllm, max_concurrent):
    """Run one RQ and save results."""
    results_file = os.path.join(output_dir, f"results_Qwen3-8B_{rq_name}.json")

    if os.path.exists(results_file):
        print(f"  {rq_name}: Results already exist at {results_file}, skipping.")
        with open(results_file) as f:
            return json.load(f)

    print(f"  {rq_name}: Querying {len(prompts)} prompts...")

    results = None
    if use_vllm:
        results = run_inference_vllm(prompts, model_path, max_concurrent)
    if results is None:
        results = run_inference_transformers(prompts, model_path, device=device)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-8B local inference for GroundLM")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to Qwen3-8B model")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-concurrent", type=int, default=8, help="Concurrency for vLLM")
    parser.add_argument("--no-vllm", action="store_true", help="Skip vLLM, use transformers")
    parser.add_argument("--output-dir", default="./groundlm_output_v2", help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    use_vllm = not args.no_vllm

    # Step 1: Generate prompts
    print("=" * 60)
    print("Step 1: Generating prompts...")
    print("=" * 60)
    all_prompts = generate_all_prompts(output_dir)

    # Step 2: Run inference for each RQ
    print("\n" + "=" * 60)
    print("Step 2: Running Qwen3-8B inference (non-thinking mode)...")
    print("=" * 60)

    all_results = {}
    for rq_name in ["format_stability", "order_sensitivity", "isomorphism_consistency", "distractor_robustness"]:
        prompts = all_prompts[rq_name]
        results = run_rq(prompts, rq_name, args.model_path, output_dir, args.device, use_vllm, args.max_concurrent)
        all_results[rq_name] = results

    # Step 3: Compute accuracy
    print("\n" + "=" * 60)
    print("Step 3: Computing task accuracy...")
    print("=" * 60)

    correct = 0
    total = 0
    for r in all_results["format_stability"]:
        if r.get("format") == "edge_list" and r.get("ordering") == "canonical":
            total += 1
            gt = str(r.get("ground_truth", "")).strip().lower()
            ans = extract_answer(r.get("model_answer", ""), r.get("task", ""), r.get("ground_truth", ""))
            if str(ans).strip().lower() == gt:
                correct += 1
    accuracy = correct / max(total, 1)
    print(f"  Task accuracy (edge_list, canonical): {accuracy:.4f} ({correct}/{total})")

    # Step 4: Compute faithfulness metrics
    print("\n" + "=" * 60)
    print("Step 4: Computing faithfulness metrics...")
    print("=" * 60)

    metrics = {}
    metrics["FS"] = compute_format_stability(all_results["format_stability"])
    metrics["OS"] = compute_order_sensitivity(all_results["order_sensitivity"])
    metrics["IC"] = compute_isomorphism_consistency(all_results["isomorphism_consistency"])
    metrics["DR"] = compute_distractor_robustness(all_results["distractor_robustness"])

    # Print main result table
    print("\n" + "=" * 80)
    print("MAIN RESULT TABLE — Qwen3-8B (non-thinking)")
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
    print(f"  {'Qwen3-8B':<25} {accuracy:>10.4f} {fs:>7.4f}({fs_ci[0]:.2f}-{fs_ci[1]:.2f}) {os_m:>7.4f}({os_ci[0]:.2f}-{os_ci[1]:.2f}) {ic_n:>7.4f}({ic_n_ci[0]:.2f}-{ic_n_ci[1]:.2f}) {ic_m:>7.4f}({ic_m_ci[0]:.2f}-{ic_m_ci[1]:.2f}) {dr:>7.4f}({dr_ci[0]:.2f}-{dr_ci[1]:.2f})")

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

    # Per-distractor
    print("\n--- Distractor Robustness by Distractor Type ---")
    for dtype, val in sorted(metrics["DR"]["DR_per_distractor"].items()):
        print(f"  {dtype}: {val:.4f}")

    # Save metrics
    summary = {
        "model": "Qwen3-8B",
        "mode": "non-thinking",
        "accuracy": accuracy,
        "FS": fs,
        "OS": os_m,
        "IC_neutral": ic_n,
        "IC_misleading": ic_m,
        "DR": dr,
        "FS_ci": list(fs_ci),
        "OS_ci": list(os_ci),
        "IC_neutral_ci": list(ic_n_ci),
        "IC_misleading_ci": list(ic_m_ci),
        "DR_ci": list(dr_ci),
        "FS_per_task": metrics["FS"]["FS_per_task"],
        "OS_per_task": metrics["OS"]["OS_per_task"],
        "IC_neutral_per_task": metrics["IC"]["IC_neutral_per_task"],
        "IC_misleading_per_task": metrics["IC"]["IC_misleading_per_task"],
        "DR_per_task": metrics["DR"]["DR_per_task"],
        "FS_per_pair": metrics["FS"]["FS_per_pair"],
        "OS_per_ordering": metrics["OS"]["OS_per_ordering"],
        "DR_per_distractor": metrics["DR"]["DR_per_distractor"],
    }
    summary_path = os.path.join(output_dir, "metrics_Qwen3-8B.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {summary_path}")


if __name__ == "__main__":
    main()
