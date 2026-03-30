#!/usr/bin/env python3
"""
qwen-vc-router Training Script
Base: Qwen 3.5 4B → Fine-tuned VC screening router
Hardware: 2x H100 80GB on E2E Networks
Method: bf16 LoRA SFT (15K samples) then quick eval
"""

import os
import json
import time
from datetime import datetime

print("=" * 60)
print("QWEN-VC-ROUTER MODEL TRAINING")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Step 1: Check GPUs ──────────────────────────────────────
print("\n[1/7] Checking GPUs...")
import torch
n_gpus = torch.cuda.device_count()
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name} ({mem:.0f} GB)")
print(f"  Total: {n_gpus} GPUs, {n_gpus * 80}GB VRAM")
assert n_gpus >= 1, "No GPU found!"

# ── Step 2: Load model ──────────────────────────────────────
print("\n[2/7] Loading Qwen 3.5 4B in bf16 (NOT 4-bit — Unsloth warns against QLoRA for Qwen 3.5)...")
t0 = time.time()

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=2048,
    dtype=torch.bfloat16,     # explicit bf16 — H100 native
    load_in_4bit=False,       # full precision, not quantized
)

print(f"  Base model loaded in {time.time() - t0:.0f}s")

# Apply LoRA — r=32 for quality on H100 (we have VRAM headroom)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# print_trainable_parameters returns None, prints to stdout
model.print_trainable_parameters()

# ── Step 3: Prepare dataset ─────────────────────────────────
print("\n[3/7] Loading and formatting datasets...")

from datasets import load_dataset, Dataset

VC_SYSTEM = (
    "You are an investment screening agent at a top-tier VC fund. "
    "Evaluate companies with extreme rigor. Distribution moats > technology moats. "
    "Capital efficiency is paramount. Self-reported benchmarks are marketing until verified. "
    "Output ONLY a JSON object: "
    '{"signal": "INVEST|LEAN_INVEST|NEUTRAL|LEAN_PASS|PASS", '
    '"conviction": 0-100, "summary": "one decisive sentence"}'
)

all_samples = []

# ── Dataset 1: Glaive function calling (open, no auth needed) ────
print("  Loading glaive-function-calling-v2 (5K samples)...")
try:
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    ds = ds.shuffle(seed=42).select(range(5000))
    count = 0
    for row in ds:
        try:
            system_text = row.get("system", "")
            chat_text = row.get("chat", "")
            if not chat_text:
                continue
            msgs = [{"role": "system", "content": system_text}] if system_text else []
            # Parse "USER: ... \n\nA: ..." format
            parts = chat_text.split("USER: ")
            for part in parts[1:]:  # skip empty first
                if "ASSISTANT:" in part:
                    user_part, assistant_part = part.split("ASSISTANT:", 1)
                    msgs.append({"role": "user", "content": user_part.strip()})
                    msgs.append({"role": "assistant", "content": assistant_part.strip()})
                elif "\n\nA:" in part:
                    user_part, assistant_part = part.split("\n\nA:", 1)
                    msgs.append({"role": "user", "content": user_part.strip()})
                    msgs.append({"role": "assistant", "content": assistant_part.strip()})
            if any(m["role"] == "assistant" for m in msgs):
                all_samples.append({"messages": msgs})
                count += 1
        except Exception:
            continue
    print(f"    → {count} samples")
except Exception as e:
    print(f"    → FAILED: {e}")

# ── Dataset 2: DeepFinance (financial reasoning) ─────────────
print("  Loading Agentar-DeepFinance-100K (5K samples)...")
try:
    ds = load_dataset("antgroup/Agentar-DeepFinance-100K", split="train")
    ds = ds.shuffle(seed=42).select(range(5000))
    count = 0
    for row in ds:
        try:
            # DeepFinance has instruction/input/output or similar
            msgs = []
            if "messages" in row and row["messages"]:
                msgs = row["messages"]
            elif "instruction" in row:
                user_content = row["instruction"]
                if row.get("input"):
                    user_content += "\n" + row["input"]
                msgs = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": row.get("output", "")},
                ]
            elif "question" in row:
                msgs = [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row.get("answer", row.get("response", ""))},
                ]
            if msgs and len(msgs) >= 2:
                # Ensure system prompt for financial context
                if msgs[0]["role"] != "system":
                    msgs.insert(0, {"role": "system", "content": "You are an expert financial analyst. Reason step by step."})
                all_samples.append({"messages": msgs})
                count += 1
        except Exception:
            continue
    print(f"    → {count} samples")
except Exception as e:
    print(f"    → FAILED: {e}")

# ── Dataset 3: Bespoke-Stratos (reasoning CoT) ──────────────
print("  Loading Bespoke-Stratos-17k (5K samples)...")
try:
    ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    ds = ds.shuffle(seed=42).select(range(5000))
    count = 0
    for row in ds:
        try:
            if "conversations" in row and row["conversations"]:
                msgs = []
                for turn in row["conversations"]:
                    role = turn.get("from", turn.get("role", ""))
                    content = turn.get("value", turn.get("content", ""))
                    if role in ("human", "user"):
                        msgs.append({"role": "user", "content": content})
                    elif role in ("gpt", "assistant"):
                        msgs.append({"role": "assistant", "content": content})
                    elif role == "system":
                        msgs.append({"role": "system", "content": content})
                if msgs and any(m["role"] == "assistant" for m in msgs):
                    all_samples.append({"messages": msgs})
                    count += 1
            elif "messages" in row and row["messages"]:
                all_samples.append({"messages": row["messages"]})
                count += 1
        except Exception:
            continue
    print(f"    → {count} samples")
except Exception as e:
    print(f"    → FAILED: {e}")

print(f"\n  Total samples: {len(all_samples)}")

# Sanity check — verify message format
valid = 0
for s in all_samples:
    msgs = s.get("messages", [])
    has_user = any(m.get("role") == "user" for m in msgs)
    has_assistant = any(m.get("role") == "assistant" for m in msgs)
    if has_user and has_assistant:
        valid += 1
print(f"  Valid (user+assistant): {valid}/{len(all_samples)}")
all_samples = [s for s in all_samples if
               any(m.get("role") == "user" for m in s["messages"]) and
               any(m.get("role") == "assistant" for m in s["messages"])]
print(f"  After filtering: {len(all_samples)}")

train_dataset = Dataset.from_list(all_samples)

# ── Step 4: SFT Training ───────────────────────────────────
print("\n[4/7] Starting SFT training...")
print(f"  Batch size: 4 per GPU x {n_gpus} GPUs x 4 grad accum = {4 * n_gpus * 4} effective")
print(f"  Epochs: 1")
print(f"  LR: 2e-4 cosine")
t_sft = time.time()

from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=8,    # bigger batch — H100 has headroom
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    logging_steps=25,
    save_strategy="no",
    max_seq_length=1024,              # shorter = 4x faster per step
    packing=False,
    seed=42,
    report_to="none",
    dataloader_num_workers=4,
)

def formatting_func(examples):
    """Convert messages list to chat template strings. Must return list[str]."""
    outputs = []
    messages_list = examples.get("messages", [])
    # Handle both batched (list of lists) and single
    if messages_list and isinstance(messages_list[0], dict):
        # Single example
        messages_list = [messages_list]
    for msgs in messages_list:
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            outputs.append(text)
        except Exception:
            outputs.append("")
    return outputs

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    formatting_func=formatting_func,
    args=sft_config,
)

print("  Training started...")
result = trainer.train()
sft_time = time.time() - t_sft
print(f"\n  SFT COMPLETE in {sft_time / 60:.1f} min")
print(f"  Final loss: {result.training_loss:.4f}")
print(f"  Steps: {result.global_step}")

# ── Step 5: GRPO Reinforcement Learning ─────────────────────
print("\n[5/8] GRPO reinforcement learning (128 rollouts per prompt)...")
t_grpo = time.time()

from trl import GRPOTrainer, GRPOConfig
import re as _re

# GRPO reward function — TRL expects (completions: list[str], **kwargs) -> list[float]
def vc_reward_fn(completions, **kwargs):
    """Reward function for VC screening. Higher = better structured output."""
    rewards = []
    for text in completions:
        if not isinstance(text, str):
            text = str(text)
        score = 0.0

        json_match = _re.search(r'\{[^{}]*"signal"[^{}]*\}', text)
        if json_match:
            score += 0.3
            try:
                parsed = json.loads(json_match.group())
                if parsed.get("signal") in ("INVEST", "LEAN_INVEST", "NEUTRAL", "LEAN_PASS", "PASS"):
                    score += 0.2
                conv = parsed.get("conviction", -1)
                if isinstance(conv, (int, float)) and 0 <= conv <= 100:
                    score += 0.2
                if parsed.get("summary") and len(str(parsed["summary"])) > 10:
                    score += 0.15
                if len(text) < 500:
                    score += 0.15
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            score = -0.5

        rewards.append(score)
    return rewards

# Build GRPO dataset — GRPOTrainer expects "prompt" column as string
screening_scenarios = [
    "AI infrastructure startup, $5M ARR, 20 employees, Series A at $50M",
    "Consumer social app, 100K DAU, no revenue, $10M seed, solo founder",
    "Enterprise SaaS, $20M ARR, 150% NRR, raised $30M Series B",
    "Crypto exchange, $50M volume, regulatory risk, 3 co-founders from Goldman",
    "Climate tech, pre-revenue, PhD founders from MIT, $2M grant funding",
    "Indian fintech, 10Cr ARR, 50 employees, Series A at 200Cr valuation",
    "Dev tools startup, open-source, 10K GitHub stars, $0 revenue, 5 engineers",
    "Healthcare AI, FDA pending, $8M raised, 2 year runway, ex-Google founders",
    "EdTech in SEA, $3M ARR, 500K users, marketplace model, 30% margins",
    "Robotics startup, $15M Series A, demo only, no production units, 40 engineers",
    "Food delivery clone in Africa, $1M GMV, heavy burn, competitive market",
    "Cybersecurity, $12M ARR, growing 200% YoY, 80% gross margins, 60 employees",
    "Real estate tech, $500K MRR, marketplace, CAC payback 18 months",
    "Vertical AI for legal, $2M ARR, 50 enterprise clients, ex-BigLaw founders",
    "Gaming studio, 1 hit game with 5M downloads, no second game, 20 people",
]

grpo_prompts = []
for scenario in screening_scenarios:
    msgs = [
        {"role": "system", "content": VC_SYSTEM},
        {"role": "user", "content": f"Quick screen: {scenario}"},
    ]
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    grpo_prompts.append({"prompt": prompt_text})

grpo_prompts = grpo_prompts * 10  # 150 prompts
grpo_dataset = Dataset.from_list(grpo_prompts)

try:
    grpo_config = GRPOConfig(
        output_dir="./grpo_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_completion_length=256,
        num_generations=64,       # 64 rollouts — safe for 2x H100
        max_prompt_length=512,
        report_to="none",
        seed=42,
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[vc_reward_fn],
        train_dataset=grpo_dataset,
        args=grpo_config,
    )

    print("  GRPO training started (64 rollouts per prompt, 150 prompts)...")
    grpo_result = grpo_trainer.train()
    grpo_time = time.time() - t_grpo
    print(f"\n  GRPO COMPLETE in {grpo_time / 60:.1f} min")
except Exception as e:
    grpo_time = time.time() - t_grpo
    print(f"\n  GRPO failed after {grpo_time / 60:.1f} min: {e}")
    import traceback; traceback.print_exc()
    print("  Continuing with SFT model only...")

# ── Step 6: Save model ──────────────────────────────────────
print("\n[6/8] Saving LoRA adapter + merged model...")
t_save = time.time()

# Save LoRA adapter
model.save_pretrained("./vc_router_lora")
tokenizer.save_pretrained("./vc_router_lora")
print(f"  LoRA adapter saved to ./vc_router_lora")

# Merge and save full model for GGUF export
print("  Merging LoRA into base model...")
model.save_pretrained_merged(
    "./vc_router_merged",
    tokenizer,
    save_method="merged_16bit",
)
print(f"  Merged model saved to ./vc_router_merged")
print(f"  Save time: {time.time() - t_save:.0f}s")

# ── Step 6: Export GGUF for Ollama ──────────────────────────
print("\n[7/8] Exporting to GGUF (Q8_0 for quality)...")
t_gguf = time.time()
try:
    model.save_pretrained_gguf(
        "./vc_router_gguf",
        tokenizer,
        quantization_method="q8_0",
    )
    print(f"  GGUF exported in {time.time() - t_gguf:.0f}s")
except Exception as e:
    print(f"  GGUF export failed (can do locally): {e}")

# ── Step 7: Quick eval ──────────────────────────────────────
print("\n[8/8] Quick evaluation on test companies...")
FastLanguageModel.for_inference(model)

test_cases = [
    ("Sarvam AI", "Indian sovereign AI, $250M raise at $1.5B, 114 employees, $3.5M revenue, 430x multiple"),
    ("Stripe", "Payments infrastructure, $91B valuation, profitable, 8000+ employees, category leader"),
    ("No-name startup", "No traction, no funding, solo founder, just an idea on a napkin"),
    ("Figma", "Design tool, was acquired by Adobe for $20B, deal cancelled, now independent at $12.5B"),
    ("IT shop", "10-person outsourcing consultancy in Pune, ₹5Cr revenue, no product, no funding"),
]

print()
for name, desc in test_cases:
    messages = [
        {"role": "system", "content": VC_SYSTEM},
        {"role": "user", "content": f"Quick screen: {name} — {desc}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to("cuda:0")

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
        )
    response = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"  {name:<20} → {response[:150]}")

# ── Summary ─────────────────────────────────────────────────
total = time.time() - t0
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"  Total wall time:  {total / 60:.1f} min")
print(f"  SFT time:         {sft_time / 60:.1f} min")
print(f"  Training samples: {len(all_samples)}")
print(f"  Final loss:       {result.training_loss:.4f}")
print(f"  GPUs:             {n_gpus}x H100 80GB")
print(f"  Model:            Qwen 3.5 4B + LoRA r=32")
print(f"  LoRA adapter:     ./vc_router_lora")
print(f"  Merged model:     ./vc_router_merged")
print(f"  GGUF:             ./vc_router_gguf")
print("=" * 60)
print()
print("Download to Mac:")
print("  scp -i ~/.ssh/<key> -r root@<gpu-host>:~/vc_router_gguf ./")
print()
print("Run locally with Ollama:")
print("  ollama create qwen-vc-router -f Modelfile")
print()
print("REMEMBER: Terminate the E2E instance to stop billing!")
