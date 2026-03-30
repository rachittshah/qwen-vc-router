#!/usr/bin/env python3
"""SFT only — save model immediately. Then GRPO runs separately."""
import os, json, time, torch
from datetime import datetime

# Don't set CUDA_VISIBLE_DEVICES — let accelerate handle GPU assignment

print("=" * 60)
print(f"SFT TRAINING — {datetime.now()}")
print("=" * 60)

from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B", max_seq_length=2048,
    dtype=torch.bfloat16, load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=64, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load datasets
all_samples = []

# Glaive FC
ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train").shuffle(seed=42).select(range(5000))
for row in ds:
    try:
        sys_text = row.get("system", "")
        chat = row.get("chat", "")
        if not chat: continue
        msgs = [{"role": "system", "content": sys_text}] if sys_text else []
        parts = chat.split("USER: ")
        for part in parts[1:]:
            for sep in ["ASSISTANT:", "\n\nA:"]:
                if sep in part:
                    u, a = part.split(sep, 1)
                    msgs.append({"role": "user", "content": u.strip()})
                    msgs.append({"role": "assistant", "content": a.strip()})
                    break
        if any(m["role"] == "assistant" for m in msgs):
            all_samples.append({"messages": msgs})
    except: pass
print(f"Glaive: {len(all_samples)}")

# DeepFinance
n = len(all_samples)
ds = load_dataset("antgroup/Agentar-DeepFinance-100K", split="train").shuffle(seed=42).select(range(5000))
for row in ds:
    try:
        if "messages" in row and row["messages"]:
            all_samples.append({"messages": row["messages"]})
        elif "instruction" in row:
            u = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
            all_samples.append({"messages": [{"role":"user","content":u},{"role":"assistant","content":row.get("output","")}]})
    except: pass
print(f"DeepFinance: {len(all_samples) - n}")

# Stratos
n = len(all_samples)
ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train").shuffle(seed=42).select(range(5000))
for row in ds:
    try:
        if "conversations" in row and row["conversations"]:
            msgs = []
            for t in row["conversations"]:
                r = t.get("from", t.get("role", ""))
                c = t.get("value", t.get("content", ""))
                if r in ("human","user"): msgs.append({"role":"user","content":c})
                elif r in ("gpt","assistant"): msgs.append({"role":"assistant","content":c})
                elif r == "system": msgs.append({"role":"system","content":c})
            if any(m["role"]=="assistant" for m in msgs):
                all_samples.append({"messages": msgs})
    except: pass
print(f"Stratos: {len(all_samples) - n}")

# Filter valid
all_samples = [s for s in all_samples if
    any(m["role"]=="user" for m in s["messages"]) and
    any(m["role"]=="assistant" for m in s["messages"])]
print(f"Total valid: {len(all_samples)}")

def formatting_func(examples):
    outputs = []
    msgs_list = examples.get("messages", [])
    if msgs_list and isinstance(msgs_list[0], dict): msgs_list = [msgs_list]
    for msgs in msgs_list:
        try: outputs.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        except: outputs.append("")
    return outputs

# Train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=Dataset.from_list(all_samples),
    formatting_func=formatting_func,
    args=SFTConfig(
        output_dir="./sft_output", per_device_train_batch_size=8,
        gradient_accumulation_steps=2, num_train_epochs=1,
        learning_rate=2e-4, optim="adamw_torch", lr_scheduler_type="cosine",
        warmup_steps=10, bf16=True, logging_steps=25, save_strategy="no",
        max_seq_length=1024, packing=False, seed=42, report_to="none",
    ),
)

t0 = time.time()
result = trainer.train()
print(f"\nSFT DONE in {(time.time()-t0)/60:.1f}min | loss={result.training_loss:.4f}")

# SAVE IMMEDIATELY
print("Saving LoRA adapter...")
model.save_pretrained("/root/vc_router_lora")
tokenizer.save_pretrained("/root/vc_router_lora")
print("Saving merged model...")
model.save_pretrained_merged("/root/vc_router_merged", tokenizer, save_method="merged_16bit")
print("SAVED. SFT complete.")
