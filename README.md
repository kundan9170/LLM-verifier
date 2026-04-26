# LLM-Verifier: Inference-Time Early-Exit for Reasoning LLMs

> An inference-time verifier pipeline that monitors chain-of-thought (CoT) reasoning and triggers early exit when reasoning is complete — saving **25–34% tokens** on math benchmarks with a frozen Qwen3-4B model.

**Undergraduate Project (UGP)** · IIT Kanpur · Under the guidance of *Prof. Sayak Ray Chowdhury*

Authors: Ruthvik Tunuguntla · Kundan Kumar · Ansh Agarwal

---

## Motivation

Reasoning-tuned LLMs (Qwen3, DeepSeek-R1) routinely emit thousands of CoT tokens — even after they have effectively solved the problem. This *overthinking* directly translates to wasted compute, higher latency, and inflated inference cost.

**Goal:** detect when a reasoning chain has produced enough information to safely terminate, without significantly degrading task accuracy — *without retraining the base model*.

---

## Results

| Benchmark | Baseline Acc. | Verifier Acc. | Baseline Tokens | Verifier Tokens | Token Savings |
|-----------|:-------------:|:-------------:|:---------------:|:---------------:|:-------------:|
| GSM8K     | 95.2%         | 89.0%         | 823             | 613             | **−25.5%**    |
| MATH-500  | 94.4%         | 91.2%         | 1024            | 745             | **−27.2%**    |
| AIME 2024 | 90.0%         | 76.7%         | 954             | 693             | **−27.4%**    |

Token savings are remarkably consistent (~25–27%) across difficulty levels; accuracy cost scales with problem complexity.

---

## Architecture

A **two-phase generation pipeline** with shared KV-cache:

```
   Question
      │
      ▼
┌──────────────────────┐         ┌────────────────────┐
│  PHASE 1: Thinking   │  ──►    │  Verifier (judge)  │
│  Qwen3-4B inside     │  ◄──    │  Qwen3-0.6B        │
│  <think>...</think>  │         │  → continue/exit/  │
│  KV-cache reuse      │         │    abort           │
└──────────────────────┘         └────────────────────┘
      │ (</think> injected on exit)
      ▼
┌──────────────────────┐
│  PHASE 2: Answering  │
│  Run to EOS,         │
│  no verification     │
│  Extract \boxed{}    │
└──────────────────────┘
```

- **Generator:** `Qwen/Qwen3-4B` (native `<think>`/`</think>` support)
- **Judge:** `Qwen/Qwen3-0.6B` with `enable_thinking=False` for speed
- **Chunking:** sentence-aware (10–60 tokens), not fixed-interval
- **KV-cache:** persists across both phases — zero prompt re-encoding

---

## Verifier Designs

We explored three verifier families of increasing sophistication:

1. **Rule-Based Verifier** — keyword matching for completion (`therefore`, `the answer is`) and self-correction (`wait`, `mistake`) cues. *Too rigid; misses deep structural errors.*
2. **Hybrid Verifier** — multi-signal: deterministic math (regex + Python recalc), n-gram loop detection (math-aware preprocessing), rolling-logprob confidence. *Brittle on multi-step reasoning.*
3. **LLM-as-a-Judge** — small companion model (Qwen3-0.6B) reads each new sentence chunk and grades it as `correct` / `wrong` / `continue`. **Final design used in benchmarks.**

We also conducted a substantial **negative-result study** on internal model signals (entropy, JSD, embedding distance, hidden-state norms, `</think>` token rank) — none gave a reliable, problem-agnostic, hard-codable signal of correctness. See the report for details.

---

## Repository Layout

```
.
├── llm_engine/              # Model loading, CoT generation, KV-cache mgmt
│   ├── model_loader.py
│   ├── cot_generator.py
│   └── final_answer_generator.py
├── verifier/                # Pluggable verifiers
│   └── rule_based_verifier.py
├── pipeline/                # Orchestration
│   ├── controller.py
│   ├── early_exit_logic.py
│   └── utils.py
├── inference/               # Benchmark drivers
├── tests/                   # Unit tests with mock model/tokenizer
├── config/
│   └── config.yaml
├── run_pipeline.py          # CLI entrypoint
└── README.md
```

---

## Setup

```bash
git clone https://github.com/kundan9170/LLM-verifier.git
cd LLM-verifier

# Recommended: virtualenv
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, HuggingFace Transformers, PyYAML, NumPy.

GPU strongly recommended — the 4B generator runs comfortably on a single 16 GB GPU in fp16/bf16.

---

## Quick Start

Run the pipeline on a single question:

```bash
python run_pipeline.py --question "What is 17 * 23 + 8?"
```

With a custom config:

```bash
python run_pipeline.py \
    --question "If a triangle has sides 3, 4, 5, what is its area?" \
    --config config/config.yaml
```

---

## Configuration

Edit `config/config.yaml`:

```yaml
llm:
  model_name: "Qwen/Qwen3-4B"
  load_dtype: "bfloat16"
  device: "auto"
  max_cot_tokens: 1280
  checkpoint_interval: 30
  return_logprobs: true

generation:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  max_new_tokens: 256

pipeline:
  enable_early_exit: true
  exit_on_confident_verdict: true
  max_verifier_calls: 50

debug:
  print_partial_cot: false
  print_verifier_decision: true
  print_final_answer: true
```

---

## Running the Tests

```bash
python -m unittest tests/test_pipeline.py
```

Tests use a `DummyModel`/`DummyTokenizer` harness so they run without GPU or model downloads.

---

## Key Engineering Lessons

- **KV-cache reuse is a correctness issue, not just an optimization.** An O(n²) recomputation bug from missing `past_key_values` between successive token generations can silently invert performance.
- **EOS handling is model-specific.** Qwen2.5 has two EOS IDs (`{151643, 151645}`); hardcoding a single one causes silent generation failures.
- **State-ordering bugs are easy to miss.** `prev_chunks.append()` running *before* `_check_repetition()` produced 100% false self-overlap.
- **Math-aware preprocessing is necessary** before n-gram comparison — strip LaTeX, operators, numbers, single-letter variables to avoid false positives on algebraic text.
- **Model-native features beat workarounds.** Switching to a model with native thinking tokens eliminated an entire `FinalAnswerGenerator` subsystem.

---

## Limitations

- Tested only on Qwen3 models and math-heavy benchmarks (GSM8K, MATH-500, AIME 2024).
- Static early-exit policy — thresholds are not per-problem adaptive.
- Judge is used zero-shot; a small fine-tune on labelled correctness data would likely close the AIME accuracy gap.

---

## Future Work

- Adaptive exit thresholds based on predicted problem difficulty.
- Problem-type routing (different verifier policies per prompt class).
- Reward-model verifier trained explicitly on step-level correctness.
- Learned linear probe on hidden-state features as a cheap pre-filter before invoking the LLM judge.

---

## Acknowledgements

We thank **Prof. Sayak Ray Chowdhury** for ongoing guidance throughout this project, and the Department of Computer Science and Engineering at IIT Kanpur for compute support.

---

## License

This project is released for academic and research use. See `LICENSE` for details.
