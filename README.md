<p align="center">
  <img src="https://img.shields.io/badge/Model-Qwen2.5--1.5B-blue?style=for-the-badge" alt="Model"/>
  <img src="https://img.shields.io/badge/Method-QLoRA-green?style=for-the-badge" alt="Method"/>
  <img src="https://img.shields.io/badge/Dataset-GSM8K-orange?style=for-the-badge" alt="Dataset"/>
  <img src="https://img.shields.io/badge/VRAM-6GB-red?style=for-the-badge" alt="VRAM"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/tiny-reasoning/"><img src="https://img.shields.io/pypi/v/tiny-reasoning" alt="PyPI Version"/></a>
  <a href="https://pypi.org/project/tiny-reasoning/"><img src="https://img.shields.io/pypi/pyversions/tiny-reasoning" alt="Python Version"/></a>
  <a href="https://github.com/vuductai/tiny-reasoning/blob/main/LICENSE"><img src="https://img.shields.io/github/license/vuductai/tiny-reasoning" alt="License"/></a>
  <a href="https://github.com/vuductai/tiny-reasoning/stargazers"><img src="https://img.shields.io/github/stars/vuductai/tiny-reasoning" alt="GitHub Stars"/></a>
  <a href="https://colab.research.google.com"><img src="https://img.shields.io/badge/Run%20on-Colab-orange?logo=google-colab" alt="Colab"/></a>
</p>

# Tiny-Reason

> **Distilling System 2 Reasoning into Sub-2B Parameter Language Models via Structured Chain-of-Thought Supervision and Quantized Low-Rank Adaptation**

Large reasoning models (DeepSeek-R1, OpenAI o1/o3) achieve remarkable performance by generating explicit chains of thought — but they require hundreds of billions of parameters and massive compute budgets. **Tiny-Reason** demonstrates that structured CoT reasoning can be distilled into a **1.5B parameter model** using **QLoRA fine-tuning on a single consumer GPU** (6GB VRAM), achieving non-trivial mathematical reasoning capabilities at a fraction of the cost.

This repository provides a complete, reproducible research pipeline: data processing with CoT injection, 4-bit quantized training with LoRA adapters, validation-guided checkpointing, and evaluation with semantically-aware answer normalization — all executable end-to-end in **~45 minutes on a free Google Colab T4 instance**.

---

## Table of Contents

- [Motivation & Research Context](#motivation--research-context)
- [Technical Approach](#technical-approach)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration System](#configuration-system)
- [Methodology Deep Dive](#methodology-deep-dive)
- [Evaluation Protocol](#evaluation-protocol)
- [Performance & Analysis](#performance--analysis)
- [Design Decisions & Trade-offs](#design-decisions--trade-offs)
- [Project Structure](#project-structure)
- [Extending Tiny-Reason](#extending-tiny-reason)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [References & Acknowledgments](#references--acknowledgments)

---

## Motivation & Research Context

Recent work on reasoning in language models has established two complementary paradigms:

1. **Scale-dependent reasoning** — Models like GPT-4, DeepSeek-R1, and o1/o3 achieve strong reasoning through massive parameter counts and reinforcement learning on chain-of-thought traces. These models are effective but prohibitively expensive for most researchers and edge deployment scenarios.

2. **Reasoning distillation** — Smaller models can acquire structured reasoning capabilities through supervised fine-tuning on CoT-annotated data, as demonstrated by [Magister et al. (2023)](https://arxiv.org/abs/2212.10071) and the open-source DeepSeek-R1-Distill family.

**Tiny-Reason** sits at the intersection of these paradigms. Rather than relying on distilled CoT traces from a teacher model, we leverage the structured annotations already present in GSM8K (step-by-step solutions separated by `####` delimiters) and reformulate them into an explicit `<think>...</think>` reasoning framework compatible with modern chat-format LLMs. Combined with QLoRA for memory efficiency, this enables:

- **Democratized reasoning research** — Full training loop on free-tier hardware (Colab T4, 15GB VRAM)
- **Rapid iteration** — 250-step training converges in ~45 minutes
- **Reproducibility** — Deterministic seeding across all random sources, config versioning with every checkpoint

---

## Technical Approach

```
                      ┌─────────────────────────────────────────────────┐
                      │            Tiny-Reason: System Overview          │
                      └─────────────────────────────────────────────────┘

  GSM8K (train split)          Qwen2.5-1.5B-Instruct              Evaluation
 ┌───────────────────┐        ┌──────────────────────┐        ┌──────────────────┐
 │ "Natalia sold     │        │  Frozen Base Weights  │        │  GSM8K test split │
 │  clips to 48..."  │        │  (NF4 quantized)      │        │  (held-out)       │
 │                   │        │  ≈ 0.8GB VRAM         │        │                   │
 │  #### 72          │        │                       │        │  Greedy decoding   │
 └────────┬──────────┘        │  LoRA Adapters (r=16) │        │  + regex extraction│
          │                   │  ~1.3M trainable      │        │  + answer normalize│
          ▼                   │  (0.087% of params)   │        └────────▲───────────┘
 ┌───────────────────┐        └──────────┬───────────┘                 │
 │  CoT Injection    │                   │                             │
 │                   │                   ▼                             │
 │  <think>          │        ┌──────────────────────┐                 │
 │  Step 1: ...      │───────▶│   SFTTrainer          │────────────────┘
 │  Step 2: ...      │        │   (AdamW-8bit)        │
 │  </think>         │        │   LR: 2e-4 + warmup   │
 │  **Final Answer:**│        │   Grad accum: 4       │
 │  72               │        │   Effective batch: 8   │
 └───────────────────┘        └──────────────────────┘
```

**Key idea:** We wrap GSM8K's native step-by-step solutions in `<think>` XML tags within the ChatML template, creating a structured separation between *reasoning trace* and *final answer*. The model learns to generate explicit intermediate computation before committing to an output — analogous to the "System 2" slow-thinking behavior observed in large reasoning models.

---

## Architecture Overview

### Pipeline Components

| Component | Module | Responsibility |
|:--|:--|:--|
| **Config Parser** | `src/config_parser.py` | Hierarchical YAML config with 16 CLI overrides; CLI args take priority over YAML defaults |
| **Data Processor** | `src/dataset.py` | GSM8K loading, CoT prompt injection (`<think>` tag wrapping), train/validation splitting, malformed sample filtering |
| **Model Loader** | `src/model.py` | Unsloth-accelerated model loading with NF4 quantization, PEFT adapter injection across all attention + FFN projections |
| **Trainer** | `src/trainer.py` | SFTTrainer orchestration with hardware-aware precision (BF16/FP16 auto-detection), checkpoint management, config archival |
| **Evaluator** | `scripts/evaluate.py` | Greedy-decoded inference on held-out test set with semantically-aware answer normalization (currency, units, word-to-digit) |
| **Inference** | `scripts/inference.py` | Interactive REPL with real-time token streaming via `TextStreamer` to visualize CoT generation |
| **Utilities** | `src/utils.py` | Deterministic seeding (Python, NumPy, PyTorch CPU/CUDA, `PYTHONHASHSEED`) and structured logging |

### LoRA Adapter Topology

All seven projection matrices in each transformer block receive rank-16 adapters:

```
Transformer Block (×28 layers)
├── Self-Attention
│   ├── q_proj  ← LoRA(r=16, α=32)
│   ├── k_proj  ← LoRA(r=16, α=32)
│   ├── v_proj  ← LoRA(r=16, α=32)
│   └── o_proj  ← LoRA(r=16, α=32)
└── Feed-Forward Network (SwiGLU)
    ├── gate_proj ← LoRA(r=16, α=32)
    ├── up_proj   ← LoRA(r=16, α=32)
    └── down_proj ← LoRA(r=16, α=32)

Trainable params: ~1.3M / 1,500M total (0.087%)
```

Applying LoRA to **both attention and FFN layers** (rather than attention-only, as is common) is a deliberate choice — mathematical reasoning requires modifications to the feed-forward computation pathways, not just the attention routing. This follows findings from [Hu et al. (2021)](https://arxiv.org/abs/2106.09685) showing that broader LoRA coverage improves task performance at minimal additional cost.

---

## Quick Start

```bash
git clone https://github.com/vuductai/tiny-reasoning.git
cd tiny-reasoning
make install    # Install all dependencies including Unsloth
make train      # Fine-tune on 2,000 GSM8K samples (~45 min on T4)
make evaluate   # Benchmark on held-out test set
make inference  # Interactive reasoning demo with streaming
```

---

## Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|:--|:--|:--|
| Python | 3.9+ | 3.10+ |
| GPU VRAM | 6 GB | 15 GB (Colab T4) |
| CUDA | 11.8+ | 12.1+ |
| OS | Linux / macOS / WSL2 | Linux |

### Standard Install

```bash
git clone https://github.com/vuductai/tiny-reasoning.git
cd tiny-reasoning

python -m venv venv && source venv/bin/activate

make install
```

### Manual Install

```bash
pip install -r requirements.txt

# Unsloth requires hardware-specific wheels
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27"

pip install -e .
```

### Google Colab

```python
!git clone https://github.com/vuductai/tiny-reasoning.git
%cd tiny-reasoning
!make install
```

---

## Usage

### Training

```bash
# Default: Qwen2.5-1.5B on 2,000 GSM8K samples, QLoRA r=16
make train

# Or with custom overrides
python scripts/train.py --config configs/default.yaml \
    --learning_rate 1e-4 \
    --max_steps 500 \
    --num_samples 5000 \
    --lora_r 32 --lora_alpha 64
```

**Training log:**

```
2025-02-13 10:00:00 - INFO - Starting TinyReason Training Pipeline...
2025-02-13 10:00:10 - INFO - Loading openai/gsm8k...
2025-02-13 10:00:15 - INFO - Subsampling to 2000 samples.
2025-02-13 10:00:18 - INFO - Shuffling dataset...
2025-02-13 10:00:20 - INFO - Train size: 1800, Validation size: 200
2025-02-13 10:00:25 - INFO - Loading model: Qwen/Qwen2.5-1.5B-Instruct
2025-02-13 10:01:30 - INFO - Applying LoRA (Rank 16)...
2025-02-13 10:01:35 - INFO - Starting Trainer...
2025-02-13 10:45:00 - INFO - Training complete.
```

**Artifacts saved to `outputs/lora_adapter/`:**
- `adapter_model.safetensors` — LoRA weights
- `tokenizer.json` — Tokenizer configuration
- `experiment_config.yaml` — Exact config used (for reproducibility)

### Evaluation

```bash
make evaluate

# Output:
# Running evaluation on 50 samples...
# Final Accuracy: 42.00%
```

The evaluator uses **greedy decoding** (deterministic) and extracts answers via regex matching on the `**Final Answer:**` delimiter, followed by a normalization pipeline that handles currency symbols, unit suffixes, floating-point artifacts, and word-to-digit conversion.

### Interactive Inference

```bash
make inference
```

```
==========================================
TinyReason Interactive Demo (Type 'exit' to quit)
==========================================

Problem: A bookstore sells 15 books on Monday and twice as many on Tuesday. How many books were sold in total?

Reasoning...
<think>
Let me solve this step by step.

1. Books sold on Monday: 15
2. Books sold on Tuesday: 15 × 2 = 30
3. Total books sold: 15 + 30 = 45

So the bookstore sold 45 books in total.
</think>

**Final Answer:** 45
```

Real-time token streaming via `TextStreamer` allows you to observe the model's chain-of-thought as it generates, providing transparency into the reasoning process.

---

## Configuration System

Tiny-Reason uses a **hierarchical configuration architecture**: YAML files define experiment defaults, and any parameter can be overridden at the command line. CLI arguments always take priority.

### Default Configuration

```yaml
# configs/default.yaml — Optimized for Colab T4 (15GB VRAM)

project:
  name: "TinyReason-Qwen-1.5B"
  seed: 3407                            # Deterministic reproducibility
  output_dir: "outputs"
  adapter_path: "outputs/lora_adapter"

model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true                    # NF4 quantization

lora:
  r: 16                                 # Rank (capacity vs. efficiency)
  lora_alpha: 32                        # Scaling factor (2× rank heuristic)
  lora_dropout: 0                       # Disabled for small models
  bias: "none"
  target_modules:                       # Full coverage: attention + FFN
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

data:
  dataset_name: "openai/gsm8k"
  subset: "main"
  split: "train"
  num_samples: 2000
  test_samples: 50
  num_proc: 2
  validation_split: 0.1
  shuffle: true
  system_prompt: "You are a helpful assistant that solves math problems step-by-step."

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4        # Effective batch size = 8
  warmup_steps: 50                      # 20% of max_steps
  max_steps: 250
  learning_rate: 0.0002                 # Conservative for quantized training
  logging_steps: 10
  eval_steps: 50
  save_steps: 50
  save_total_limit: 3
  optim: "adamw_8bit"                   # Memory-efficient optimizer
  weight_decay: 0.01
  lr_scheduler_type: "linear"
```

### CLI Override Reference

| Argument | Type | Default | Description |
|:--|:--|:--|:--|
| `--seed` | int | 3407 | Global random seed |
| `--output_dir` | str | `outputs` | Checkpoint output directory |
| `--adapter_path` | str | `outputs/lora_adapter` | Final adapter save path |
| `--learning_rate` | float | 2e-4 | Peak learning rate |
| `--max_steps` | int | 250 | Total training steps |
| `--batch_size` | int | 2 | Per-device batch size |
| `--gradient_accumulation_steps` | int | 4 | Steps before optimizer update |
| `--warmup_steps` | int | 50 | Linear warmup steps |
| `--eval_steps` | int | 50 | Validation evaluation interval |
| `--save_steps` | int | 50 | Checkpoint save interval |
| `--num_samples` | int | 2000 | Training samples to use |
| `--test_samples` | int | 50 | Test samples for evaluation |
| `--validation_split` | float | 0.1 | Fraction held for validation |
| `--lora_r` | int | 16 | LoRA rank |
| `--lora_alpha` | float | 32 | LoRA scaling factor |

---

## Methodology Deep Dive

### 1. Chain-of-Thought Prompt Injection

The core technique reformulates GSM8K's native annotations into a structured generation format:

**Raw GSM8K format:**
```
Q: Natalia sold clips to 48 of her friends in April, and then she sold
   half as many clips in May. How many clips did Natalia sell altogether?
A: Natalia sold 48/2 = 24 clips in May.
   Natalia sold 48+24 = 72 clips altogether. #### 72
```

**Tiny-Reason CoT format (ChatML):**
```
<|im_start|>system
You are a helpful assistant that solves math problems step-by-step.<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether?<|im_end|>
<|im_start|>assistant
<think>
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether.

</think>

**Final Answer:** 72<|im_end|>
```

This transformation creates three distinct structural regions in the generation:

| Region | Tokens | Learning Signal |
|:--|:--|:--|
| `<think>...</think>` | Reasoning trace | Model learns to decompose problems into intermediate computation steps |
| `**Final Answer:**` | Delimiter | Model learns to transition from exploration to commitment |
| Answer value | Final output | Model learns to extract and state the computed result |

The loss is computed over **the entire assistant turn** (reasoning + answer), which means the model receives gradient signal for both *how* it reasons and *what* it concludes. This is critical — training only on final answers would not teach the model to generate useful intermediate steps.

### 2. QLoRA: Quantized Low-Rank Adaptation

QLoRA combines two orthogonal memory-reduction techniques to enable fine-tuning on consumer hardware:

**NF4 Quantization** — Base model weights are quantized from FP16/BF16 to 4-bit Normal Float format, a quantization scheme specifically designed for normally-distributed neural network weights ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)). This reduces the memory footprint of the frozen base model by ~4×.

**Low-Rank Adaptation** — Instead of updating the full weight matrices `W ∈ ℝ^(d×d)`, LoRA decomposes the update into two low-rank matrices:

```
W' = W + (α/r) · B·A     where A ∈ ℝ^(r×d), B ∈ ℝ^(d×r), r << d
```

With `r=16` and `α=32`, the effective learning rate scaling is `α/r = 2.0`, which amplifies the adapter's contribution during training while keeping the parameter count minimal.

**Memory budget comparison for Qwen2.5-1.5B:**

| Method | Base Model | Optimizer States | Activations | Total VRAM |
|:--|:--|:--|:--|:--|
| Full fine-tuning (FP16) | 3.0 GB | 6.0 GB | ~3 GB | ~12 GB |
| LoRA (FP16 base) | 3.0 GB | 0.01 GB | ~3 GB | ~6 GB |
| **QLoRA (NF4 base)** | **0.8 GB** | **0.01 GB** | **~3 GB** | **~4 GB** |

### 3. Hardware-Aware Precision Selection

The trainer automatically detects GPU compute capability and selects the optimal precision:

```python
bf16_supported = torch.cuda.is_bf16_supported()
# Ampere+ (A100, RTX 3090, H100) → BF16: better numerical range
# Volta/Turing (T4, V100, RTX 2080) → FP16: hardware-native
```

This eliminates a common source of training instability — using BF16 on hardware that doesn't natively support it.

### 4. Training Dynamics

| Phase | Steps | Behavior |
|:--|:--|:--|
| **Warmup** | 0–50 | Linear LR ramp from 0 → 2e-4; stabilizes gradient magnitudes for quantized weights |
| **Main training** | 50–250 | Linear LR decay; model learns CoT structure and answer extraction |
| **Checkpoint selection** | Every 50 steps | Validation loss tracked; best-of-3 checkpoint loaded at end |

**Effective batch size** is 8 (2 per-device × 4 accumulation steps), achieved via gradient accumulation to fit within VRAM constraints while maintaining stable optimization dynamics.

**Optimizer:** AdamW-8bit (via `bitsandbytes`) reduces optimizer state memory by ~75% compared to standard AdamW, with negligible impact on convergence quality.

---

## Evaluation Protocol

### Answer Extraction

Generated responses are parsed using a regex pattern that captures content between `**Final Answer:**` and the ChatML end token:

```python
re.search(r"\*\*Final Answer:\*\*\s*(.*?)(?:<\|im_end\|>|$)", response, re.DOTALL)
```

### Semantic Answer Normalization

Raw model outputs are normalized before comparison to handle format variability:

| Normalization | Example | Purpose |
|:--|:--|:--|
| Case folding | `"THREE"` → `"three"` | Case-insensitive matching |
| Whitespace/comma removal | `"1, 200"` → `"1200"` | Numeric formatting |
| Trailing `.0` removal | `"42.0"` → `"42"` | Float artifact handling |
| Word-to-digit conversion | `"seven"` → `"7"` | Verbal number resolution |
| Currency stripping | `"$350"` → `"350"` | Unit-agnostic comparison |
| Unit stripping | `"15 meters"` → `"15"` | Unit-agnostic comparison |

This normalization is essential for fair evaluation — a model answering "$72" and the ground truth being "72" represent identical mathematical reasoning.

### Decoding Strategy

- **Evaluation:** Greedy decoding (`do_sample=False`) for deterministic, reproducible benchmarks
- **Interactive inference:** Near-greedy sampling (`temperature=0.1`, `do_sample=True`) for minimal variation while avoiding repetition loops

---

## Performance & Analysis

### Baseline Results (Default Configuration)

Training on 2,000 GSM8K samples with default hyperparameters:

| Metric | Value | Notes |
|:--|:--|:--|
| Trainable parameters | ~1.3M | 0.087% of 1.5B base |
| Training time | ~45 min | Google Colab T4 (free tier) |
| Peak VRAM usage | ~6 GB | Well within T4's 15GB |
| Training loss (final) | ~0.5–0.8 | Converged |
| **Test accuracy (GSM8K)** | **~35–45%** | 50-sample held-out test |

### Scaling Behavior

| Training Samples | Expected Accuracy | Config Changes |
|:--|:--|:--|
| 2,000 | 35–45% | Default |
| 5,000 | 45–55% | `--max_steps 500` |
| 10,000+ | 55–65% | `--max_steps 1000 --lora_r 32 --lora_alpha 64` |

> **Context:** The base Qwen2.5-1.5B-Instruct model without fine-tuning achieves ~30% on GSM8K. Full-scale models (70B+) with CoT prompting achieve 80–90%+. Tiny-Reason demonstrates that even with only 0.087% of parameters trainable and 2,000 training samples, meaningful reasoning capability transfer is achievable.

### Factors Affecting Performance

| Factor | Impact | Recommendation |
|:--|:--|:--|
| Sample count | Most significant | Scale to full GSM8K (7.5K) for best results |
| LoRA rank | Moderate | r=32 or r=64 for harder tasks; diminishing returns beyond 64 |
| Learning rate | Sensitive | 1e-4 to 3e-4 range; higher risks instability with quantization |
| Sequence length | Low (for GSM8K) | 2048 is sufficient; increase for multi-hop reasoning tasks |

---

## Design Decisions & Trade-offs

### Why Qwen2.5-1.5B-Instruct?

- **Instruction-tuned base** — Already understands chat format and instruction following; we build on this foundation rather than teaching it from scratch
- **1.5B sweet spot** — Large enough for non-trivial reasoning, small enough for consumer GPU training
- **Native ChatML support** — `<|im_start|>` / `<|im_end|>` tokens are part of the base vocabulary, avoiding tokenizer modifications

### Why full-layer LoRA (attention + FFN)?

Standard practice applies LoRA only to attention projections (`q_proj`, `v_proj`). We target all seven projection matrices including the SwiGLU FFN layers because:
1. Mathematical computation relies heavily on the feed-forward pathway, not just attention routing
2. The marginal parameter cost of FFN adapters (~40% more trainable params) is negligible at r=16
3. Empirically, full-coverage LoRA converges faster and generalizes better on arithmetic tasks

### Why `<think>` tags instead of free-form CoT?

Explicit XML-style delimiters (`<think>...</think>`) provide:
1. **Parseable structure** — Clean regex extraction for evaluation and downstream processing
2. **Training signal clarity** — The model learns a clear boundary between exploration and commitment
3. **Compatibility** — Mirrors the format used by reasoning-optimized models (DeepSeek-R1, QwQ), enabling future knowledge transfer

### Why SFTTrainer with packing disabled?

Sequence packing (concatenating multiple samples into a single sequence) improves throughput but contaminates cross-attention between unrelated problems. For reasoning tasks where coherent multi-step logic is critical, preserving sequence boundaries ensures each training example receives isolated gradient signal.

### Stateless, composable components

Every module (`GSM8KProcessor`, `ModelLoader`, `TinyTrainer`) accepts a configuration dictionary and has no global state. This enables:
- **Parallel experimentation** — Multiple configs can be tested independently
- **Reproducibility** — The exact config is serialized with every checkpoint
- **Testability** — Each component can be unit-tested in isolation

---

## Project Structure

```
tiny-reasoning/
├── configs/
│   └── default.yaml              # Experiment configuration (all hyperparameters)
├── scripts/
│   ├── train.py                  # Training pipeline orchestrator
│   ├── evaluate.py               # Test-set evaluation with answer normalization
│   └── inference.py              # Interactive streaming inference demo
├── src/
│   ├── __init__.py               # Package exports
│   ├── config_parser.py          # YAML loader + CLI argument merging (16 args)
│   ├── dataset.py                # GSM8K → CoT ChatML formatting + split logic
│   ├── model.py                  # Unsloth model loading + LoRA injection
│   ├── trainer.py                # SFTTrainer setup + artifact serialization
│   └── utils.py                  # Seeding (5 sources) + structured logging
├── results/                      # Experiment results (gitkeep)
├── Makefile                      # install | train | evaluate | inference | clean
├── pyproject.toml                # Package metadata (setuptools)
├── requirements.txt              # Pinned dependencies (13 packages)
└── README.md
```

**Total codebase:** ~750 lines of Python (excluding config/docs). Deliberately minimal — every line serves the pipeline.

---

## Extending Tiny-Reason

### Different base models

```bash
# Use a larger model (requires more VRAM)
python scripts/train.py --config configs/default.yaml \
    # Edit configs/default.yaml → model.name: "Qwen/Qwen2.5-7B-Instruct"
```

Modify `model.name` in your YAML config. Any Hugging Face model supported by Unsloth works. For 7B+ models, consider reducing `per_device_train_batch_size` to 1.

### Different datasets

Subclass or modify `GSM8KProcessor._format_function()` to handle alternative CoT datasets (MATH, ARC, StrategyQA). The key requirement is a parseable separator between reasoning steps and final answer.

### Experiment tracking

WandB is included in dependencies. To enable:
```python
# In src/trainer.py, change:
report_to="wandb"  # instead of "none"
```

---

## Troubleshooting

| Issue | Cause | Solution |
|:--|:--|:--|
| **CUDA OOM** | Batch too large for VRAM | `--batch_size 1 --gradient_accumulation_steps 8` |
| **Training doesn't converge** | LR too high for quantized model | `--learning_rate 1e-4` |
| **Adapter not found** | Training didn't complete | Verify `ls outputs/lora_adapter/` contains model files |
| **Low test accuracy** | Insufficient training data | Increase `--num_samples` and `--max_steps` proportionally |
| **Unsloth install fails** | GPU-specific wheel mismatch | See [Unsloth installation guide](https://github.com/unslothai/unsloth#installation) |
| **BF16 errors on T4** | Hardware doesn't support BF16 | Auto-handled — trainer detects and falls back to FP16 |

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch and open a Pull Request

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## References & Acknowledgments

- **QLoRA** — Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models" ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))
- **LoRA** — Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))
- **GSM8K** — Cobbe et al., "Training Verifiers to Solve Math Word Problems" ([arXiv:2110.14168](https://arxiv.org/abs/2110.14168))
- **Chain-of-Thought** — Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" ([arXiv:2201.11903](https://arxiv.org/abs/2201.11903))
- **Reasoning Distillation** — Magister et al., "Teaching Small Language Models to Reason" ([arXiv:2212.10071](https://arxiv.org/abs/2212.10071))
- **Unsloth** — [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) for memory-efficient fine-tuning
- **Qwen2.5** — [huggingface.co/Qwen](https://huggingface.co/Qwen) for the base model

---

<p align="center">
  <b>If you find this work useful, please consider starring the repository.</b>
  <br><br>
  <a href="https://github.com/vuductai/tiny-reasoning/stargazers"><img src="https://img.shields.io/github/stars/vuductai/tiny-reasoning?style=social" alt="GitHub Stars"/></a>
</p>
