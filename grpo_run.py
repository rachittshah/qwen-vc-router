#!/usr/bin/env python3
"""GRPO on single GPU. Run 2 instances with different configs for A/B test.
Usage: CUDA_VISIBLE_DEVICES=0 python grpo_run.py --config A
       CUDA_VISIBLE_DEVICES=1 python grpo_run.py --config B
"""
import os, json, time, sys, re
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
import torch
from datasets import Dataset

config_name = sys.argv[1] if len(sys.argv) > 1 else "A"
gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
print(f"GRPO Config {config_name} on GPU {gpu_id}")

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# Load SFT model — fast_inference=False required for Qwen3.5 (DeltaNet not in vLLM)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/vc_router_lora",
    max_seq_length=1024, dtype=torch.bfloat16, load_in_4bit=False,
    fast_inference=False,
)
FastLanguageModel.for_training(model)

VC_SYSTEM = (
    "You are an investment screening agent at a top-tier VC fund. "
    "Evaluate companies with extreme rigor. Distribution moats > technology moats. "
    "Capital efficiency is paramount. Output ONLY a JSON object: "
    '{"signal": "INVEST|LEAN_INVEST|NEUTRAL|LEAN_PASS|PASS", '
    '"conviction": 0-100, "summary": "one decisive sentence"}'
)

# Reward function
def vc_reward_fn(completions, **kwargs):
    rewards = []
    for text in completions:
        if not isinstance(text, str): text = str(text)
        score = 0.0
        m = re.search(r'\{[^{}]*"signal"[^{}]*\}', text)
        if m:
            score += 0.3
            try:
                p = json.loads(m.group())
                if p.get("signal") in ("INVEST","LEAN_INVEST","NEUTRAL","LEAN_PASS","PASS"): score += 0.2
                c = p.get("conviction", -1)
                if isinstance(c,(int,float)) and 0 <= c <= 100: score += 0.2
                if p.get("summary") and len(str(p["summary"])) > 10: score += 0.15
                if len(text) < 500: score += 0.15
            except: pass
        else:
            score = -0.5
        rewards.append(score)
    return rewards

# Build prompts
scenarios = [
    "AI infrastructure startup, $5M ARR, 20 employees, Series A at $50M",
    "Consumer social app, 100K DAU, no revenue, $10M seed, solo founder",
    "Enterprise SaaS, $20M ARR, 150% NRR, raised $30M Series B",
    "Crypto exchange, $50M volume, regulatory risk, 3 co-founders from Goldman",
    "Climate tech, pre-revenue, PhD founders from MIT, $2M grant funding",
    "Indian fintech, 10Cr ARR, 50 employees, Series A at 200Cr valuation",
    "Dev tools, open-source, 10K GitHub stars, $0 revenue, 5 engineers",
    "Healthcare AI, FDA pending, $8M raised, ex-Google founders",
    "EdTech in SEA, $3M ARR, 500K users, marketplace model",
    "Robotics, $15M Series A, demo only, no production units",
    "Food delivery clone in Africa, $1M GMV, heavy burn",
    "Cybersecurity, $12M ARR, growing 200% YoY, 80% margins",
    "Real estate tech, $500K MRR, marketplace, CAC 18mo payback",
    "Vertical AI for legal, $2M ARR, 50 enterprise clients",
    "Gaming studio, 1 hit game 5M downloads, no second game",
]

prompts = []
for s in scenarios:
    msgs = [{"role":"system","content":VC_SYSTEM},{"role":"user","content":f"Quick screen: {s}"}]
    prompts.append({"prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)})
prompts = prompts * 10  # 150 total

# Config A vs B
configs = {
    "A": {"num_generations": 8, "lr": 5e-5, "max_completion_length": 256},
    "B": {"num_generations": 16, "lr": 1e-5, "max_completion_length": 256},
}
cfg = configs[config_name]

grpo_config = GRPOConfig(
    output_dir=f"./grpo_output_{config_name}",
    per_device_train_batch_size=cfg["num_generations"],
    gradient_accumulation_steps=1,   # >1 can cause NaN grads with GRPO (issue #3006)
    num_train_epochs=1,
    learning_rate=cfg["lr"],
    optim="adamw_torch",
    bf16=True, logging_steps=5, save_strategy="no",
    max_completion_length=cfg["max_completion_length"],
    num_generations=cfg["num_generations"],
    max_prompt_length=512,
    report_to="none", seed=42,
)

trainer = GRPOTrainer(
    model=model, tokenizer=tokenizer,
    reward_funcs=[vc_reward_fn],
    train_dataset=Dataset.from_list(prompts),
    args=grpo_config,
)

t0 = time.time()
print(f"GRPO {config_name}: {cfg['num_generations']} rollouts, lr={cfg['lr']}")
result = trainer.train()
print(f"\nGRPO {config_name} DONE in {(time.time()-t0)/60:.1f}min")

# Save
model.save_pretrained(f"/root/grpo_model_{config_name}")
tokenizer.save_pretrained(f"/root/grpo_model_{config_name}")
print(f"Saved to /root/grpo_model_{config_name}")
