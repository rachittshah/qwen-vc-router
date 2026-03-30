# VC Pipeline Performance Architecture: Research Synthesis

## Source: 14 Opus 4.6 Research Agents (March 2026)

Covering: agent filesystems, multi-model routing, tool use optimization, context windows, real-time data, reasoning patterns, memory systems, production infra, eval at scale, VC-specific optimization, fine-tuning, RL training, March 2026 models, eval best practices.

---

## The 10 Highest-Impact Optimizations (ROI-ordered)

### 1. Programmatic Tool Calling (PTC)
**Impact: 37-99% token reduction. The single biggest lever.**

Instead of the model selecting one tool at a time (each requiring a full inference pass), the agent writes Python that calls tools in loops inside a sandboxed container. Intermediate results stay OUT of context.

- 37% average token reduction (Anthropic internal)
- 81% on complex batch operations (Cloudflare)
- 98.7% in extreme cases (150K tokens -> 2K for MCP workflows)
- GA since Feb 2026 on Sonnet 4.6+

Enable: Add `code_execution_20260120` tool + set `allowed_callers: ["code_execution_20260120"]` on tools.

**For this pipeline:** Agent writes Python that calls `harmonic_search_companies`, filters results, calls `crm_search_knowledge`, cross-references, and returns only exceptions. 20 API calls = 1 code block instead of 20 inference passes.

### 2. Tool Search / Deferred Loading
**Impact: 85% context reduction, +25pp accuracy on Opus.**

Instead of loading all tool schemas upfront (55K tokens for 50+ tools), load on demand via semantic search.

- 85% context reduction (77K -> 8.7K tokens)
- Accuracy: Opus 4 improved 49% -> 74%
- Supports up to 10,000 tools in catalog

Enable: Set `defer_loading: true` on tools, include `tool_search_tool_bm25_20251119`. Keep 3-5 most-used tools always loaded.

**For this pipeline:** Defer Harmonic's 13 tools + the CRM's 19 tools. Load on demand. Keep WebSearch + WebFetch always loaded.

### 3. Filesystem as Extended Mind (Scratchpad Pattern)
**Impact: 39% -> 83.6% accuracy (Hindsight paper, 20B model with externalized memory vs GPT-4o full context at 60.2%)**

Agents write intermediate results to files instead of keeping them in context. Reading files is recall. Writing files is thinking.

- Anthropic's multi-agent researcher saves plans to Memory before 200K cutoff
- Cursor uses git worktrees for parallel agent isolation
- Claude Code's `claude-progress.txt` pattern

**For this pipeline:** Each agent writes findings to `vc_state/scratch/{eval_id}/{agent_id}.json`. CIO reads summaries, not full outputs. Knowledge store persists across evals.

### 4. Observation Masking Between Layers
**Impact: 52% cheaper, 2.6% MORE accurate (JetBrains, Dec 2025)**

Replace old tool outputs with placeholders, keep recent 10 turns. Beats LLM summarization on cost AND quality.

**For this pipeline:** Already implemented. Each layer passes compressed context (signal + conviction + one sentence) to the next layer.

### 5. Prompt Caching
**Impact: 90% cost reduction on cached input tokens.**

All 9 agents share the same company context. Cache it once, read at 0.1x cost.

- 35 agents x 20K shared context: $2.10 uncached -> $0.28 cached
- Cache hierarchy: tools -> system -> messages
- Minimum: 4,096 tokens for Opus, 2,048 for Sonnet

**For this pipeline:** Structure company context (CRM enrichment + quantitative score + knowledge store) as cached system prompt prefix. Agent-specific instructions as uncached suffix.

### 6. Qwen 3.5 4B as Router Model
**Impact: 50-70% API cost reduction with <5% quality loss.**

Fine-tuned Qwen 3.5 4B (97.5% tool calling accuracy, March 2, 2026) screens companies in <100ms before Opus touches anything.

- RouteLLM: 85% cost savings, 95% quality retention
- Fin-R1: SFT + GRPO on small model beats DeepSeek-R1 (671B) on financial tasks
- Training: 5-10K distilled Opus evaluations, ~$25 on rented A100

**For this pipeline:** Qwen 3.5 4B classifies INVEST/PASS at near-zero cost. Only INVEST signals trigger the $2 Opus pipeline.

### 7. Plan-then-Execute (not ReAct)
**Impact: 92% vs 85% accuracy, 4x cheaper than Reflexion.**

Separate planning from execution. The planner generates structured steps; the executor runs tools.

**For this pipeline:** Already our architecture (4-layer pipeline). Validated by research.

### 8. Structured Adversarial Debate
**Impact: Structured > unstructured > cooperative > single-agent (FinNLP EMNLP 2025)**

The 5-act protocol: Intelligence Distillation -> Adversarial Debate -> Final Synthesis.

**For this pipeline:** Our Contrarian agent + CIO synthesis is exactly this pattern. GuruAgents paper proves encoding investment philosophy into prompts generates 42.2% CAGR vs 29.4% benchmark.

### 9. Competitive Landscape as Highest-ROI Agent Task
**Impact: 83% recall, 20x speed improvement (2.5 days -> 3 hours)**

Optic's multi-agent competitive discovery: 83% recall vs 65% OpenAI Deep Research vs 60% Perplexity.

**For this pipeline:** Our Layer 1 Competitive Landscape agent. Should be the most resourced agent (more turns, more tools, more context).

### 10. DAG-Shapley for Agent Attribution
**Impact: 83.7% reduction in LLM calls for per-agent contribution scoring.**

Exploit DAG structure of agent workflows to compute exact Shapley values efficiently. Better than concordance for Darwinian weight evolution.

**For this pipeline:** Replace concordance-based weight updates with DAG-Shapley attribution after each batch run.

---

## Custom Model Architecture

### Qwen 3.5 4B Fine-Tuning Plan

**Phase 1 (Week 1, $0):** Zero-shot validation
```bash
ollama pull qwen3.5:4b
# Test on 10 companies: can it classify INVEST/PASS with structured JSON?
```

**Phase 2 (Weeks 2-3, ~$25):** LoRA fine-tune
- Base: Qwen 3.5 4B (Apache 2.0)
- Data: 5-10K distilled from Opus evaluations + VCBench labels
- Framework: Unsloth + TRL on rented A100
- Task: Company classification + structured extraction

**Phase 3 (Month 2, ~$50):** GRPO reinforcement learning
- Reward: classification accuracy (40%) + structured output (20%) + factual accuracy (20%) + reasoning quality (20%)
- Framework: TRL GRPOTrainer
- Eval: VCBench F0.5 as regression gate

**Deployment:**
```
Qwen 3.5 4B (fine-tuned, <100ms, ~$0.001/company)
  -> INVEST -> Opus 4.6 full pipeline (~$2/company)
  -> PASS -> log & skip ($0)
```

---

## Memory Architecture

### 6-Layer Compounding Memory Stack

| Layer | What | Tool | Timescale |
|-------|------|------|-----------|
| Constitution | Investment philosophy as rules | CLAUDE.md | Permanent |
| Semantic Memory | Extracted facts + relationships | Mem0-style / knowledge.py | Per-eval |
| Episodic Memory | Past interaction retrieval | SQLite + FTS5 (46 microsecond search) | Cross-session |
| Graph Memory | Investor networks, company relationships | Neo4j/Graphiti (3.4x improvement on multi-hop) | Cross-eval |
| Reflective Memory | Post-failure rule extraction | Reflexion pattern (+20% reasoning) | Post-eval |
| Forgetting | Importance-based Ebbinghaus decay | 4-tier TTL (critical/high/medium/low) | Continuous |

---

## Anti-Patterns (Research-Confirmed)

| Don't | Why | Source |
|-------|-----|--------|
| Reflection loops | +2.2% accuracy at 4.1x cost | ATLAS paper, Enterprise benchmark |
| CoT on reasoning models | +2-3% gain at 20-80% latency cost | Wharton GAIL study |
| Self-consistency beyond N=10 | Power-law: first 5 samples capture most value | Scaling research |
| Market sizing via agents | Highest hallucination risk of any VC task | Multiple sources |
| Stuffing 1M tokens | 200K optimized beats 1M stuffed (83% vs 64%) | Augment Code data |
| Tool count >30 | Selection accuracy degrades significantly | MCP tool description paper |
| Raw LLM financial numbers | 27% hallucination on earnings predictions | Financial AI survey |

---

## The Production Stack

```
[Qwen 3.5 4B Router] -- screen 10K companies/hour at ~$0
    |
    INVEST signals only
    |
[Quant Score Gate] -- quantitative filter ($0)
    |
[CRM Enrichment] -- CRM context ($0)
    |
[Knowledge Injection] -- institutional facts ($0)
    |
[Scratchpad Init] -- create eval workspace
    |
[Layer 1: Intelligence] -- 4 agents, Opus 4.6, PTC enabled
    | write findings to scratch/
    | observation mask for next layer
[Layer 2: Thesis] -- 3 agents, Opus 4.6, weighted vote
    | DAG-Shapley attribution
[Layer 3: Risk] -- 1 agent, Opus 4.6, veto power
    |
[Layer 4: CIO] -- 1 agent, Opus 4.6, reads scratch/ summaries
    |
[IC Memo Generator] -- 145-line shareable memo
    |
[CRM Write-back] -- auto-push to CRM
[Knowledge Store] -- extract and persist facts
[Darwinian Weights] -- update based on DAG-Shapley
[Langfuse] -- full trace with scores
```

---

## Eval Architecture (Research Synthesis)

### The Eval Stack (from Hamel, Eugene Yan, Prometheus, Haize Labs)

**Principle 1: Error analysis first, always.** Review 100+ real agent traces before writing evals. Categorize failures bottom-up. Count frequencies. This determines what to build. (Hamel)

**Principle 2: One evaluator per dimension.** Never a "God Judge." Separate judges for: factual accuracy, competitive completeness, market sizing reasonableness, risk identification, signal quality. (Eugene Yan)

**Principle 3: Binary pass/fail over scores.** "Is the competitive analysis factually accurate? Yes/No" beats "Rate quality 1-5." Forces clarity, prevents hedging, yields consistent labeling. (Hamel)

**Principle 4: FActScore for factual claims.** Decompose each IC memo into atomic claims, verify each against source data. ChatGPT achieves only 58% factual accuracy on bio generation — this is the biggest quality risk. (FActScore paper)

**Principle 5: PoLL ensemble for judges.** 3 smaller models beat single GPT-4 at 1/7th cost. (Eugene Yan)

**Principle 6: Prometheus 2 as open-source judge.** 0.897 Pearson correlation vs GPT-4's 0.882. Rubric-based, free to run. (KAIST)

**Principle 7: Adversarial stress testing.** Feed deliberately misleading data and measure if agents flag it. 44% attack success rate on baseline agents. (Haize Labs)

### Per-Dimension Evaluators for This Pipeline

| Dimension | Type | Method | Cost |
|-----------|------|--------|------|
| JSON schema compliance | Deterministic | Pydantic validation | $0 |
| Signal enum valid | Deterministic | Enum check | $0 |
| Conviction in range | Deterministic | Range check | $0 |
| Bull/bear case present | Deterministic | Field check | $0 |
| Factual accuracy | FActScore | Atomic claim decomposition + source verification | Medium |
| Competitive completeness | LLM judge (Prometheus 2) | Rubric: "Does the analysis mention the top 3 competitors?" | Low |
| Market sizing reasonableness | LLM judge + cross-reference | Compare against known data sources | Medium |
| Risk identification | LLM judge (Prometheus 2) | Rubric: "Does the analysis identify the top 3 risks?" | Low |
| Signal calibration | Statistical | Concordance vs VCBench ground truth | $0 |
| Thesis coherence | LLM judge | "Does the verdict match the stated evidence?" | Low |

### The Eval Flywheel

```
Agent produces output
  → Deterministic evals (instant, $0)
  → LLM judge on 10% sample (Prometheus 2, free)
  → FActScore on INVEST signals only (high-stakes verification)
  → Failures become golden set items
  → Golden set drives prompt optimization (GEPA)
  → Optimized prompts produce better outputs
  → Repeat
```
