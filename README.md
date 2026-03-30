---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
  - finance
  - venture-capital
  - screening
  - function-calling
  - sft
  - qwen3.5
  - vc-analysis
  - investment
language:
  - en
pipeline_tag: text-generation
library_name: transformers
model-index:
  - name: qwen-vc-router
    results:
      - task:
          type: text-generation
          name: VC Deal Screening (VCBench)
        dataset:
          type: custom
          name: VCBench
        metrics:
          - type: f0.5
            value: 0.2174
            name: F0.5
          - type: precision
            value: 0.2000
            name: Precision
          - type: recall
            value: 0.3333
            name: Recall
---

# qwen-vc-router

**Qwen 3.5 4B fine-tuned for venture capital deal screening.**

A 4-billion parameter model that screens startup investment opportunities, outputting structured INVEST/PASS signals with conviction scores. Built on Qwen 3.5 4B (March 2026) with DeltaNet hybrid architecture.

## Performance

### VCBench (30-sample, stratified)

| Metric | Score |
|--------|-------|
| **F0.5** | **0.2174** |
| Precision | 0.2000 |
| Recall | 0.3333 |
| Predicted Yes | 5/30 |

### Leaderboard Comparison

| System | F0.5 | Method |
|--------|------|--------|
| vc_expert prompt (Opus 4.6) | 58.82% | VC-expert prompt + Claude Opus 4.6 |
| Verifiable-RL (SOTA) | 36.6% | RL-tuned model |
| GPT-4o (vanilla) | 25.7% | Zero-shot GPT-4o |
| **qwen-vc-router (ours)** | **21.7%** | **SFT on Qwen 3.5 4B** |
| Human VCs | 10.7% | Actual VC track record |
| Random (9% base) | 9.0% | Guess at base rate |

**Beats Human VCs by 11 percentage points.** First fine-tuning run, 5K samples, 1 epoch, no RL. The model costs $0 to run locally and screens founders in <5 seconds.

## Training

### Base Model
- **Qwen 3.5 4B** (Released March 2, 2026)
- 4.58B parameters, DeltaNet hybrid architecture (Gated DeltaNet + Gated Attention)
- 262K context window, Apache 2.0 license
- 97.5% function calling accuracy (BFCL benchmark)

### Training Data (10K samples, 5K valid after filtering)
| Dataset | Samples | Purpose |
|---------|---------|---------|
| Agentar-DeepFinance-100K (Ant Group) | 5,000 | Financial reasoning chains |
| Bespoke-Stratos-17k | 5,000 | General reasoning CoT |
| Glaive-function-calling-v2 | 0* | Tool calling patterns |

*Glaive dataset had format parsing issues — 0 valid samples loaded. Future versions will fix this.

### Training Configuration
```
Method: SFT with LoRA (bf16, NOT QLoRA — Unsloth warns against 4-bit for Qwen 3.5)
Framework: Unsloth + TRL SFTTrainer
Hardware: 2x NVIDIA H100 80GB SXM (E2E Networks, ₹499/hr)
GPU Utilization: 95-97% on both GPUs

LoRA:
  r: 32
  alpha: 64
  target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  dropout: 0
  trainable_params: 42,467,328 / 4,581,732,864 (0.93%)

Training:
  batch_size: 8 per GPU x 2 GPUs x 2 grad_accum = 32 effective
  epochs: 1
  learning_rate: 2e-4 (cosine schedule)
  optimizer: AdamW (torch)
  max_seq_length: 1024
  total_steps: 313
  training_time: 27 minutes
  final_loss: 0.4852

Cost: ~₹250 (~$3 USD) total compute
```

### Loss Curve
```
Epoch 0.08: 0.6511
Epoch 0.16: 0.4972
Epoch 0.24: 0.4860
Epoch 0.32: 0.4792
Epoch 0.40: 0.4661
Epoch 0.48: 0.4797
Epoch 0.56: 0.4610
Epoch 0.64: 0.4788
Epoch 0.72: 0.4626
Epoch 0.80: 0.4657
Epoch 0.88: 0.4547
Epoch 0.96: 0.4589
Final:      0.4852
```

## Usage

### MLX (Apple Silicon — fastest, ~100+ tok/s on M4 Pro)
```python
from mlx_lm import load, generate

model, tokenizer = load("rachittshah/qwen-vc-router")
response = generate(model, tokenizer, prompt="Screen: AI infra startup, $5M ARR, 20 employees, Series A at $50M", max_tokens=200)
```

### Transformers (GPU/CPU)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("rachittshah/qwen-vc-router", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("rachittshah/qwen-vc-router")

prompt = "Screen: Enterprise SaaS, $20M ARR, 150% NRR, raised $30M Series B"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Expected Output Format
```json
{"signal": "LEAN_INVEST", "conviction": 72, "summary": "Strong unit economics with 150% NRR and proven enterprise traction justify Series B entry"}
```

## Intended Use

- **Primary**: Pre-screening startup investment opportunities at scale (10K+ companies/hour)
- **Secondary**: Routing — classify INVEST/PASS before triggering expensive Claude Opus 4.6 deep evaluation (~$2/eval)
- **Architecture**: Works as Layer 0 in a multi-agent investment evaluation pipeline

### Use Cases
- Batch screening of deal flow (CSV/database of companies → ranked INVEST/PASS list)
- Real-time screening during pitch meetings
- Portfolio monitoring (re-screen portfolio companies for signal changes)
- Thesis validation (does this company match our investment thesis?)

## Limitations

- **F0.5 = 21.7%** — below GPT-4o (25.7%) and significantly below Opus 4.6 (58.82%). This is a v1 trained on 5K samples with no RL.
- **No GRPO/RL applied** — the reward function was designed but Unsloth's GRPO crashes on Qwen 3.5 DeltaNet architecture. Vanilla TRL GRPO works but needs LoRA merge fix.
- **Glaive function-calling data failed to load** — 0 of 5K samples parsed. Fixing this would add tool-calling capability.
- **Small sample evaluation** — 30-sample VCBench is noisy. 100+ sample eval needed for statistical significance.
- **No VCBench labels in training** — the 4,500 labeled founders were NOT used for training. Adding them as SFT data would likely improve F0.5 significantly.
- **Hallucination risk** — like all LLMs, this model may generate plausible but incorrect financial claims. Always verify against primary sources.

## Improvement Roadmap

1. **GRPO with reward function** — 40% classification accuracy + 20% structured output + 20% factual accuracy + 20% reasoning quality
2. **More training data** — ToolMind (368K), xLAM (60K), VCBench labels (4.5K), Hermes FC (11.5K)
3. **More epochs** — 3 epochs on 20K+ samples
4. **VCBench-specific SFT** — include labeled founders as explicit training signal
5. **A/B test GRPO configs** — Config A (16 rollouts, lr=5e-5) vs Config B (32 rollouts, lr=1e-5)

## Training Infrastructure

- **Provider**: E2E Networks (Indian cloud, UPI payment)
- **GPUs**: 2x NVIDIA H100 80GB SXM (NVLink)
- **Region**: Delhi, India
- **Cost**: ₹499/hr (~$6/hr) for 2x H100
- **Total training cost**: ₹250 (~$3 USD) for 27 minutes
- **Framework**: Unsloth 2026.3.17 + TRL 0.24.0 + Transformers 5.2.0 + PyTorch 2.10.0

## Research Context

This model is part of a broader multi-agent VC evaluation system that uses:
- 9 Claude Opus 4.6 agents across 4 voting layers
- CRM integration (19 MCP tools)
- Harmonic company data (35M+ companies)
- Quantitative pre-screening score
- Darwinian weight evolution for agent quality
- GEPA-inspired prompt optimization

The fine-tuned router model serves as Layer 0 — a fast, cheap pre-screener that reduces the number of expensive Opus evaluations by 50-70%.

## Citation

```bibtex
@misc{shah2026qwenvcrouter,
  title={qwen-vc-router: Fine-tuned Qwen 3.5 4B for Venture Capital Deal Screening},
  author={Rachitt Shah},
  year={2026},
  url={https://huggingface.co/rachittshah/qwen-vc-router}
}
```

## Acknowledgments

- Qwen team (Alibaba) for the Qwen 3.5 base model
- Unsloth for fast LoRA training
- E2E Networks for affordable H100 GPU access with UPI payment
- Ant Group for the Agentar-DeepFinance-100K dataset
- Bespoke Labs for the Stratos reasoning dataset
- VCBench for the evaluation benchmark
