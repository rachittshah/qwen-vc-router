#!/usr/bin/env python3
"""GRPO using vanilla TRL (no Unsloth) — Qwen3.5 DeltaNet breaks Unsloth's compiled GRPO."""
import os, json, time, sys, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

config_name = sys.argv[1] if len(sys.argv) > 1 else "A"
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
print(f"GRPO {config_name} on GPU {gpu} (vanilla TRL, no Unsloth)")

# Load base model + LoRA adapter directly (bypass Unsloth)
print("Loading model...")
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "/root/vc_router_lora")
# Don't merge — keep LoRA trainable for GRPO weight updates
model.base_model.warnings_issued = {"estimate_tokens": True}
model.warnings_issued = {"estimate_tokens": True}
tokenizer = AutoTokenizer.from_pretrained("/root/vc_router_lora")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Loaded in {time.time()-t0:.0f}s")

VC_SYSTEM = (
    "You are an investment screening agent at a top-tier VC fund. "
    "Output ONLY JSON: "
    '{"signal": "INVEST|LEAN_INVEST|NEUTRAL|LEAN_PASS|PASS", '
    '"conviction": 0-100, "summary": "one sentence"}'
)

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
                if isinstance(c,(int,float)) and 0<=c<=100: score += 0.2
                if p.get("summary") and len(str(p["summary"]))>10: score += 0.15
                if len(text)<500: score += 0.15
            except: pass
        else:
            score = -0.5
        rewards.append(score)
    return rewards

scenarios = [
    "AI infra, $5M ARR, 20 employees, Series A at $50M",
    "Consumer social, 100K DAU, no revenue, $10M seed",
    "Enterprise SaaS, $20M ARR, 150% NRR, $30M Series B",
    "Crypto exchange, $50M volume, regulatory risk",
    "Climate tech, pre-revenue, MIT PhD founders",
    "Indian fintech, 10Cr ARR, 50 employees",
    "Dev tools, 10K GitHub stars, $0 revenue",
    "Healthcare AI, FDA pending, $8M raised",
    "EdTech SEA, $3M ARR, 500K users",
    "Robotics, $15M Series A, demo only",
    "Cybersecurity, $12M ARR, 200% YoY growth",
    "Vertical AI legal, $2M ARR, 50 enterprise clients",
]

prompts = []
for s in scenarios:
    msgs = [{"role":"system","content":VC_SYSTEM},{"role":"user","content":f"Screen: {s}"}]
    prompts.append({"prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)})
prompts = prompts * 8  # 96 prompts

configs = {
    "A": {"num_generations": 4, "lr": 5e-5},
    "B": {"num_generations": 8, "lr": 1e-5},
}
cfg = configs[config_name]

config = GRPOConfig(
    output_dir=f"./grpo_{config_name}",
    per_device_train_batch_size=cfg["num_generations"],
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=cfg["lr"],
    optim="adamw_torch",
    bf16=True,
    logging_steps=5,
    save_strategy="no",
    max_completion_length=200,
    num_generations=cfg["num_generations"],
    max_prompt_length=300,
    report_to="none",
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[vc_reward_fn],
    train_dataset=Dataset.from_list(prompts),
    args=config,
)

print(f"GRPO {config_name}: {cfg['num_generations']} rollouts, lr={cfg['lr']}, 96 prompts")
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"\nGRPO {config_name} DONE in {elapsed/60:.1f}min")

model.save_pretrained(f"/root/grpo_model_{config_name}")
tokenizer.save_pretrained(f"/root/grpo_model_{config_name}")
print(f"Saved to /root/grpo_model_{config_name}")
