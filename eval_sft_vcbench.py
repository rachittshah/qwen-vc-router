#!/usr/bin/env python3
"""
VCBench Evaluation for qwen-vc-router SFT Model
Full benchmark: 4,500 founders, F0.5 metric, compare to Opus baseline

Methodology matches VCBench paper (Vela Partners):
- Binary classification: will founder build a successful startup?
- F0.5 weights precision 2x over recall (bad investment > missed deal)
- Stratified sampling preserves 9% base rate
- Compare to: Opus 4.6 (58.82%), Verifiable-RL (36.6%), GPT-4o (25.7%), Human VCs (10.7%)
"""

import json
import csv
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# MLX for fast Apple Silicon inference (80-120+ tok/s on M4 Pro)
from mlx_lm import load, generate

# Try our SFT model first, fall back to pre-quantized base
MODEL_OPTIONS = [
    "models/vc_router_merged",                # Our SFT model
    "mlx-community/Qwen3.5-4B-OptiQ-4bit",         # Pre-quantized (fastest, 3GB)
]

VCBENCH_DATA = Path("benchmark/data/vcbench_final_public.csv")
RESULTS_DIR = Path("benchmark/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MODEL_NAME = "qwen-vc-router-sft"

# Load model
print("Loading model via MLX...")
_model = None
_tokenizer = None
for model_path in MODEL_OPTIONS:
    try:
        print(f"  Trying: {model_path}")
        _model, _tokenizer = load(model_path)
        MODEL_NAME = model_path.split("/")[-1]
        print(f"  Loaded: {MODEL_NAME}")
        break
    except Exception as e:
        print(f"  Failed: {e}")
        continue

if _model is None:
    print("ERROR: No model loaded")
    sys.exit(1)

def model_query(prompt, max_tokens=150):
    """Query via MLX — fast Apple Silicon inference."""
    try:
        response = generate(
            _model, _tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return response.strip()
    except Exception as e:
        return f"ERROR: {e}"


def load_vcbench(path, sample_size=None, seed=42):
    """Load VCBench dataset with stratified sampling."""
    import random
    random.seed(seed)

    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if sample_size and sample_size < len(rows):
        positives = [r for r in rows if r.get("success", "0") == "1"]
        negatives = [r for r in rows if r.get("success", "0") != "1"]

        # Stratified: preserve base rate
        n_pos = max(1, round(sample_size * len(positives) / len(rows)))
        n_neg = sample_size - n_pos

        sampled_pos = random.sample(positives, min(n_pos, len(positives)))
        sampled_neg = random.sample(negatives, min(n_neg, len(negatives)))
        rows = sampled_pos + sampled_neg
        random.shuffle(rows)

    return rows


def build_prompt(founder_data):
    """Build VCBench prompt from founder data — matches paper methodology."""
    # Extract available fields
    industry = founder_data.get("industry", "Unknown")
    education = founder_data.get("educations_json", "")
    jobs = founder_data.get("jobs_json", "")
    prose = founder_data.get("anonymised_prose", "")

    prompt = f"""Evaluate this founder for startup success potential.

Industry: {industry}
Education: {education[:500]}
Work History: {jobs[:500]}
Profile: {prose[:500]}

Will this founder build a successful startup (unicorn exit, IPO, or major acquisition)?
Answer with ONLY "YES" or "NO" followed by a one-sentence reason."""

    return prompt


def parse_prediction(response):
    """Parse YES/NO from model response."""
    response_upper = response.upper().strip()
    if response_upper.startswith("YES"):
        return 1
    elif response_upper.startswith("NO"):
        return 0
    # Check for YES/NO anywhere in first 50 chars
    first_50 = response_upper[:50]
    if "YES" in first_50:
        return 1
    if "NO" in first_50:
        return 0
    return 0  # default to NO (conservative)


def compute_f05(tp, fp, fn):
    """Compute F0.5 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    return f05


def main():
    print("=" * 60)
    print("VCBENCH EVALUATION: qwen-vc-router SFT Model")
    print(f"Model: {MODEL_NAME}")
    print(f"Sample: {SAMPLE_SIZE} founders")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Verify model loaded
    print(f"\n[1/4] Checking model...")
    test = model_query("Say hello in one word.")
    print(f"  Model OK: {test[:80]}...")

    # Load data
    print(f"\n[2/4] Loading VCBench ({SAMPLE_SIZE} founders, stratified)...")
    if not VCBENCH_DATA.exists():
        print(f"  ERROR: {VCBENCH_DATA} not found")
        sys.exit(1)

    founders = load_vcbench(VCBENCH_DATA, SAMPLE_SIZE)
    n_positive = sum(1 for f in founders if f.get("success", "0") == "1")
    n_negative = len(founders) - n_positive
    base_rate = n_positive / len(founders) if founders else 0
    print(f"  Loaded {len(founders)} founders ({n_positive} positive, {n_negative} negative)")
    print(f"  Base rate: {base_rate:.1%}")

    # Run evaluation
    print(f"\n[3/4] Evaluating {len(founders)} founders...")
    predictions = []
    actuals = []
    t0 = time.time()

    for i, founder in enumerate(founders):
        prompt = build_prompt(founder)
        response = model_query(prompt)
        pred = parse_prediction(response)
        actual = 1 if founder.get("success", "0") == "1" else 0

        predictions.append(pred)
        actuals.append(actual)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(founders) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(founders)} | {rate:.1f} founders/sec | ETA: {eta:.0f}s")

    eval_time = time.time() - t0

    # Compute metrics
    print(f"\n[4/4] Computing metrics...")
    tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
    fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)
    tn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = compute_f05(tp, fp, fn)
    accuracy = (tp + tn) / len(predictions) if predictions else 0
    predicted_yes = sum(predictions)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  F0.5:              {f05:.4f}")
    print(f"  Precision:         {precision:.4f}")
    print(f"  Recall:            {recall:.4f}")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Predicted Yes:     {predicted_yes}/{len(predictions)}")
    print(f"  True Positives:    {tp}")
    print(f"  False Positives:   {fp}")
    print(f"  False Negatives:   {fn}")
    print(f"  True Negatives:    {tn}")
    print(f"  Eval time:         {eval_time:.0f}s ({len(predictions)/eval_time:.1f} founders/sec)")

    # Leaderboard comparison
    print("\n  LEADERBOARD:")
    baselines = [
        ("vc_expert prompt (Opus 4.6)", 0.5882),
        ("Verifiable-RL (prev SOTA)", 0.366),
        ("GPT-4o (vanilla)", 0.257),
        ("Human VCs", 0.107),
        ("Random (9% base)", 0.09),
    ]
    for name, score in baselines:
        delta = f05 - score
        status = "BEATS" if delta > 0 else "BEHIND"
        print(f"    {name:<30} {score:.4f}  {delta:+.4f}  {status}")

    print(f"\n  SFT Router (ours):            {f05:.4f}")

    # Save results
    result = {
        "model": MODEL_NAME,
        "base_model": "Qwen3.5-4B",
        "training": "SFT on DeepFinance-5K + Stratos-5K + Glaive-FC",
        "sample_size": SAMPLE_SIZE,
        "base_rate": base_rate,
        "f05": f05,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "predicted_yes": predicted_yes,
        "eval_time_seconds": eval_time,
        "timestamp": datetime.now().isoformat(),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }

    result_file = RESULTS_DIR / f"vcbench_sft_router_{SAMPLE_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_file}")

    print("\n" + "=" * 60)
    if f05 > 0.5882:
        print("  NEW RECORD! SFT Router beats Opus 4.6 baseline!")
    elif f05 > 0.366:
        print("  Beats Verifiable-RL SOTA. Competitive result for a 4B model.")
    elif f05 > 0.257:
        print("  Beats GPT-4o vanilla. Decent for a 4B fine-tuned model.")
    elif f05 > 0:
        print("  Some signal but below GPT-4o. Needs more training data or GRPO.")
    else:
        print("  No signal. Model predicts all NO or all YES.")
    print("=" * 60)


if __name__ == "__main__":
    main()
