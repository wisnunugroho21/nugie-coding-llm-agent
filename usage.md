# Usage Tutorial — Kimi-Linear (GDN-2) Code-Generation LLM

A complete, step-by-step guide to installing, training, evaluating, and extending
this project. It assumes no prior familiarity with the codebase — only that you can
run Python from a terminal.

For the *why* behind the architecture, read [`README.md`](README.md). This document
is the *how*.

---

## Table of contents

1. [What you are running](#1-what-you-are-running)
2. [Requirements & installation](#2-requirements--installation)
3. [Quick start (2 minutes)](#3-quick-start-2-minutes)
4. [The two presets: `tiny` vs `small`](#4-the-two-presets-tiny-vs-small)
5. [The full workflow, step by step](#5-the-full-workflow-step-by-step)
6. [Understanding the outputs (`runs/`)](#6-understanding-the-outputs-runs)
7. [Two-phase training: pretrain → SFT](#7-two-phase-training-pretrain--sft)
8. [Evaluation explained: perplexity & `pass@k`](#8-evaluation-explained-perplexity--passk)
9. [Code execution & security](#9-code-execution--security)
10. [Using the project as a library (Python API)](#10-using-the-project-as-a-library-python-api)
11. [Configuration reference](#11-configuration-reference)
12. [Scaling & precision](#12-scaling--precision)
13. [Troubleshooting](#13-troubleshooting)
14. [Repository layout](#14-repository-layout)

---

## 1. What you are running

A **decoder-only language model that generates Python code**, built from scratch in
JAX / Flax NNX. It follows the **Kimi Linear** hybrid-attention architecture but uses
**Gated DeltaNet-2** as the linear token mixer. Around the model sits a complete
training and evaluation cycle (the `codegen/` package): a byte-level BPE tokenizer,
the MBPP data pipeline, the optimization loop, a sampler, and a **functional
`pass@k` evaluator that actually executes generated code against unit tests**.

There are two ways to drive it:

* **Command line** — `python -m codegen.<module>` for tokenizer → train → evaluate.
* **Interactively** — the [`main.ipynb`](main.ipynb) notebook walks the whole cycle
  with explanations, and [`smoke_test.py`](smoke_test.py) runs it as one script.

---

## 2. Requirements & installation

### 2.1 Python and JAX

* **Python 3.10+** is recommended.
* The heavy dependency is **JAX**. You must install the JAX build that matches your
  hardware — this is the one install step the `requirements.txt` cannot decide for you.

```bash
# 1. clone / enter the project
cd nugie-coding-llm-agent

# 2. (recommended) create an isolated environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. install the dependencies
pip install -r requirements.txt
```

`requirements.txt` pins plain CPU `jax`/`jaxlib`. For an NVIDIA GPU, install the CUDA
build **instead**:

```bash
pip install -U "jax[cuda12]"      # NVIDIA GPU (CUDA 12)
# CPU-only is already covered by requirements.txt
```

### 2.2 What gets installed

| Package | Why |
|---|---|
| `jax`, `jaxlib`, `flax`, `optax` | the model, autodiff, and optimizer |
| `orbax-checkpoint` | save/restore model state |
| `numpy` | host-side tensors in the data pipeline |
| `datasets` | downloads MBPP the first time you run |
| `tokenizers` | the byte-level BPE backend |
| `tqdm` | progress bars |

The first run downloads the **MBPP** dataset (~a few hundred problems); after that it
is cached by `datasets` and runs offline.

### 2.3 A note on platforms

* **Linux / macOS**: the `pass@k` sandbox enforces POSIX CPU and memory `rlimit`s in
  addition to a wall-clock timeout.
* **Windows**: `rlimit`s are not available, so only the **wall-clock timeout** applies
  during `pass@k`. Everything else works identically. Because `pass@k` runs generated
  code, prefer a container or VM on Windows if you ever evaluate an untrusted model
  (see [§9](#9-code-execution--security)).

---

## 3. Quick start (2 minutes)

The fastest way to confirm everything works is the end-to-end **smoke test** on the
`tiny` CPU preset. It trains a tokenizer, trains a tiny model, evaluates perplexity,
then samples and **executes** code for a handful of MBPP problems:

```bash
python smoke_test.py            # full cycle (~2 min on a laptop CPU)
python smoke_test.py --steps 40 # shorter
python smoke_test.py --steps 120 --problems 5
```

A successful run ends with:

```
SMOKE TEST PASSED — every stage ran end-to-end.
```

> The `tiny` model is intentionally too small to solve MBPP. **`pass@k ≈ 0` is the
> expected, healthy result** — the test proves the *plumbing*, not model quality.

**Prefer it interactive?** Open [`main.ipynb`](main.ipynb) in Jupyter/VS Code and
*Run All*. It is the same cycle with narration, intermediate inspection (formatted
prompts, loss masks, parameter counts), and a "scaling up" section at the end.

---

## 4. The two presets: `tiny` vs `small`

Every run is fully described by one `TrainConfig`. Two presets are provided in
[`codegen/config.py`](codegen/config.py):

| | **`tiny`** | **`small`** |
|---|---|---|
| Purpose | CPU smoke / plumbing check | real (modest) single-GPU run |
| `d_model` | 128 | 512 |
| `n_layers` | 4 | 12 |
| Experts (top-k) | 4 (top-2) | 8 (top-2) |
| MoE schedule | 1 MLA every 4 layers | 1 MLA every 4 layers (3:1 GDN-2:MLA) |
| `vocab_size` | 2048 | 16000 |
| `train_seq_len` | 288 | 256 |
| MBPP split | `sanitized` (~120 train) | `full` (374 train) |
| Total params | ~a few M | ~200M (far fewer *active* per token) |
| Where it runs | laptop CPU, minutes | a single GPU |

Pick the preset with `--preset tiny|small` on every CLI command. Individual fields are
overridable with flags (see below) or by editing the config in Python.

---

## 5. The full workflow, step by step

The cycle is **three commands**: train the tokenizer, train the model, evaluate. They
share the same preset so paths line up automatically under `runs/<preset>/`.

### Step 1 — Train the byte-level BPE tokenizer

```bash
python -m codegen.tokenizer --preset small
```

* Trains a byte-level BPE on the MBPP **train/validation/prompt** splits only (never
  the test split — that would leak test code into the vocabulary).
* Byte-level means **no `<unk>`** and indentation is preserved exactly — both critical
  for code.
* Saves to `runs/small/tokenizer.json`.

Flags: `--vocab-size N` (override target vocab), `--out PATH` (override save path).

### Step 2 — Train the model

```bash
python -m codegen.train --preset small
```

What the loop does (all already wired in):

* **Objective** — SFT-style masked next-token cross-entropy. Each MBPP task is framed
  as a prompt (description + the asserts, which reveal the function signature) followed
  by the reference solution and `<|endoftext|>`. **Loss applies only to the solution
  span** — the prompt and padding are masked out, so the model learns *task → code*,
  not to echo the prompt.
* **Optimizer** — AdamW with a warmup→cosine learning-rate schedule, gradient clipping,
  and weight decay applied only to ≥2-D kernels (not norms/biases).
* **Gradient accumulation** — effective batch = `batch_size × grad_accum`.
* **MoE load balancing** — two mechanisms run automatically: the Switch/DeepSeek softmax
  **aux loss** (added to the loss) and the **aux-loss-free** per-expert router-bias
  nudge toward uniform load after every step (a non-trainable variable, kept out of the
  gradient).
* **Periodic eval & checkpointing** — held-out perplexity every `eval_every` steps;
  the lowest-perplexity model is saved to `ckpt-best`, and `ckpt-last` tracks the most
  recent step.

Common overrides (all optional; defaults come from the preset):

```bash
python -m codegen.train --preset small \
    --epochs 60 --lr 2e-4 --grad-accum 4 \
    --batch-size 16 --out-dir runs/exp1 --seed 0
```

| Flag | Meaning |
|---|---|
| `--epochs N` | passes over the MBPP train split |
| `--max-steps N` | hard step cap (overrides `--epochs`) |
| `--batch-size N` | sequences per forward (micro-batch) |
| `--grad-accum N` | micro-batches per optimizer step |
| `--lr F` | peak learning rate |
| `--out-dir DIR` | where checkpoints/logs land |
| `--seed N` | RNG seed |
| `--pretrain-*` | optional Phase-1 pretraining (see [§7](#7-two-phase-training-pretrain--sft)) |

### Step 3 — Evaluate (perplexity + functional `pass@k`)

`pass@k` executes generated code, so it is **gated behind an environment variable**:

```bash
# Linux / macOS
CODEGEN_ALLOW_EXEC=1 \
python -m codegen.evaluate --preset small --ckpt runs/small/ckpt-best
```

```powershell
# Windows (PowerShell)
$env:CODEGEN_ALLOW_EXEC = "1"
python -m codegen.evaluate --preset small --ckpt runs/small/ckpt-best
```

Without `CODEGEN_ALLOW_EXEC=1`, evaluation still reports **perplexity** but `pass@k`
is reported as 0 (code is not run). Useful flags:

| Flag | Meaning |
|---|---|
| `--ckpt DIR` | **required** — checkpoint directory to load |
| `--preset tiny\|small` | must match how the model was trained |
| `--split test` | MBPP split to score (default `test`) |
| `--max-problems N` | limit number of problems (faster) |
| `--n-samples N` | samples per problem (more → tighter `pass@k`) |
| `--temperature F` | sampling temperature |
| `--no-passk` | perplexity only, skip execution |

Example — a quick functional check on 100 problems with 10 samples each:

```bash
CODEGEN_ALLOW_EXEC=1 python -m codegen.evaluate --preset small \
    --ckpt runs/small/ckpt-best --max-problems 100 --n-samples 10 --temperature 0.4
```

The evaluator also prints one generated sample with its execution status for a
qualitative look.

---

## 6. Understanding the outputs (`runs/`)

Everything a run produces lands under `runs/<preset>/` (the whole `runs/` directory is
git-ignored):

```
runs/small/
  tokenizer.json      # the trained BPE tokenizer (step 1)
  config.json         # the full resolved TrainConfig + param count (provenance)
  train_log.jsonl     # one JSON record per logged step: ce, ppl, aux, lr, seq/s, val_ppl
  ckpt-best/          # lowest held-out perplexity (use this for evaluation)
  ckpt-last/          # most recent step (for resuming/inspection)
  ckpt-pretrain/      # only if you ran Phase-1 pretraining (§7)
```

* `train_log.jsonl` is append-only and flushed as it goes — tail it to watch training,
  or load it with pandas for plots.
* A checkpoint stores only the model **state** (the pytree of parameters/variables). On
  load, the module structure is rebuilt abstractly with `nnx.eval_shape` and the arrays
  are restored into it — so you must load with the **same config/preset** you trained.

---

## 7. Two-phase training: pretrain → SFT

MBPP alone is tiny (374 train problems). For stronger models, run an optional **Phase-1
plain-LM pretrain** on a larger Python corpus *before* the MBPP SFT phase. Both phases
run in one `train` invocation when you pass `--pretrain-corpus`.

In Phase 1 the corpus is packed into `train_seq_len` blocks and trained with a standard
all-token next-token loss (the loss mask is all-ones), using its own warmup→cosine
schedule. It saves `ckpt-pretrain`, then Phase 2 continues those weights as MBPP SFT.

**Two corpus sources are supported:**

```bash
# (a) a LOCAL directory of .py files
python -m codegen.train --preset small \
    --pretrain-corpus /path/to/python/repo --pretrain-epochs 1

# (b) a streamed HuggingFace code dataset (select the text column with --pretrain-hf-*)
python -m codegen.train --preset small \
    --pretrain-corpus codeparrot/codeparrot-clean-valid \
    --pretrain-hf-field content --pretrain-max-docs 20000
```

| Flag | Meaning |
|---|---|
| `--pretrain-corpus X` | local directory **or** HuggingFace dataset id (enables Phase 1) |
| `--pretrain-epochs N` | passes over the packed corpus |
| `--pretrain-max-steps N` | step cap (overrides epochs) |
| `--pretrain-max-docs N` | cap on documents read (keeps the corpus in RAM) |
| `--pretrain-lr F` | Phase-1 peak LR (defaults to `--lr`) |
| `--pretrain-hf-field` | text column for a HF dataset (default `content`) |
| `--pretrain-hf-name` | HF config/name |
| `--pretrain-hf-data-dir` | HF `data_dir` (e.g. `data/python`) |

**Tip:** for best token efficiency, train the BPE tokenizer on the same corpus you
pretrain on. The corpus is held in RAM (capped by `--pretrain-max-docs`); for very
large corpora you would stream-tokenize to a memmap (a small change in
`data.pack_lm_dataset`).

---

## 8. Evaluation explained: perplexity & `pass@k`

Two complementary metrics:

* **Perplexity** (cheap, no execution) — `exp` of the masked next-token cross-entropy
  on the held-out **completion** tokens. Lower is better. Computed during training (to
  pick `ckpt-best`) and at eval time.
* **Functional `pass@k`** (the real signal) — for each test task, sample `n_samples`
  programs, **execute** each against the task's `assert` tests, count how many pass
  (`c`), and report the unbiased estimator

  ```
  pass@k = 1 − C(n−c, k) / C(n, k)        (Chen et al., 2021)
  ```

  averaged over all tasks. `pass@1` ≈ "does one sample work"; `pass@5` ≈ "does at least
  one of five work". The preset's `eval_ks` controls which k's are reported.

The report also includes **execution status counts** — how many runs ended in
`passed` / `failed` / `timeout` / `error` / `disabled` — which tells you whether low
scores are wrong answers vs. crashes vs. runaway generations.

---

## 9. Code execution & security

`pass@k` runs **model-generated code**. The sandbox in
[`codegen/sandbox.py`](codegen/sandbox.py) is *pragmatic, not airtight*:

* each candidate runs in a **fresh child process** (Python isolated mode, `-I`) so it
  cannot corrupt the trainer;
* a hard **wall-clock timeout** kills infinite loops / runaway generations;
* on POSIX, **CPU-seconds and address-space `rlimit`s** are also enforced;
* it runs in a **throwaway temp directory** with a minimal environment.

It does **not** sandbox network or filesystem access. Therefore:

* Execution is **opt-in**: it only runs when `CODEGEN_ALLOW_EXEC=1`. Without the flag,
  `run_unit_test` returns status `disabled` and `pass@k` is 0.
* For your **own** trained models on MBPP (as in this project) that is acceptable.
* For **untrusted** models, or at scale, run evaluation inside a container / gVisor /
  firejail / VM.

---

## 10. Using the project as a library (Python API)

You don't have to use the CLI. Every stage is a plain function. This mirrors what
[`smoke_test.py`](smoke_test.py) and [`main.ipynb`](main.ipynb) do:

```python
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # CPU; drop for GPU
os.environ["CODEGEN_ALLOW_EXEC"] = "1"          # allow pass@k to execute code

import flax.nnx as nnx
from codegen.config import get_preset
from codegen.tokenizer import CodeTokenizer, train_tokenizer, _mbpp_corpus
from codegen.data import load_sft_datasets, load_eval_problems
from codegen.train import train
from codegen.checkpointing import load_checkpoint
from codegen.evaluate import evaluate_perplexity, evaluate_pass_at_k
from codegen.sampling import generate_completions
from codegen.evaluate import STOP_STRINGS

# 1. config (shrink for a fast demo)
cfg = get_preset("tiny")
cfg.max_steps = 80

# 2. tokenizer
train_tokenizer(_mbpp_corpus(cfg), cfg.vocab_size, save_path=cfg.tokenizer_path)
tok = CodeTokenizer.load(cfg.tokenizer_path)

# 3. train -> returns the best checkpoint path
best_ckpt = train(cfg)

# 4. reload + held-out perplexity
cfg.model.vocab_size = tok.vocab_size
model = load_checkpoint(cfg.model, best_ckpt)
_, val = load_sft_datasets(cfg, tok)
print("perplexity:", evaluate_perplexity(model, val, cfg.batch_size))

# 5. generate code for one problem
prob = load_eval_problems(cfg, split="test", max_problems=1)[0]
print(generate_completions(model, tok, prob.prompt, n_samples=1,
                           stops=STOP_STRINGS, max_new_tokens=128)[0])

# 6. functional pass@k
print(evaluate_pass_at_k(model, tok, cfg, verbose=False)["pass_at_k"])
```

Handy entry points:

| Function | Purpose |
|---|---|
| `config.get_preset(name)` | a ready-made `TrainConfig` |
| `tokenizer.train_tokenizer(corpus, vocab, save_path)` | fit the BPE |
| `tokenizer.CodeTokenizer.load/encode/decode` | tokenize text |
| `data.load_sft_datasets(cfg, tok)` | `(train, val)` masked token datasets |
| `data.load_eval_problems(cfg, split, max_problems)` | MBPP eval problems (prompt + tests) |
| `train.train(cfg)` | full loop → best-checkpoint path |
| `checkpointing.load_checkpoint(cfg.model, path)` | rebuild a model from a checkpoint |
| `sampling.generate_completions(...)` | prompt-in → decoded completions |
| `evaluate.evaluate_perplexity / evaluate_pass_at_k` | the two metrics |

---

## 11. Configuration reference

The most useful `TrainConfig` knobs (full list in
[`codegen/config.py`](codegen/config.py)). Set them via CLI flags or directly on the
config object in Python.

**Model** (nested under `cfg.model`, a `KimiLinearConfig`): `d_model`, `n_layers`,
`full_attn_period` (1 MLA layer every N; rest are GDN-2), `gdn_*` (linear-mixer head
dims and `gdn_chunk_size`), `mla_*` (full-attention head dims), `moe_*` (experts,
`moe_top_k`, FFN width), `max_seq_len`, `vocab_size` (auto-set from the tokenizer at
load time).

**Data / tokenizer:** `vocab_size`, `tokenizer_path`, `mbpp_config` (`full` |
`sanitized`), `train_seq_len` (**must be a multiple of `gdn_chunk_size`** — the GDN-2
chunkwise core reshapes `L` into `L/C` chunks).

**Optimization:** `batch_size`, `grad_accum`, `epochs`, `max_steps`, `lr`,
`min_lr_ratio`, `warmup_ratio`, `weight_decay`, `grad_clip`, `adam_b1`, `adam_b2`,
`router_bias_lr` (the aux-loss-free MoE balancing step size).

**Logging / checkpointing:** `out_dir`, `log_every`, `eval_every`, `passk_every`
(0 = only at the end), `ckpt_every`, `keep_ckpts`, `seed`.

**Evaluation:** `eval_max_problems` (0 = all), `eval_n_samples`, `eval_ks`,
`eval_temperature`, `eval_top_p`, `eval_max_new_tokens`, `eval_timeout` (seconds per
unit-test run).

**Precision:** `matmul_precision` (`"highest"` fp32 | `"high"` | `"bfloat16"`) — see
[§12](#12-scaling--precision).

---

## 12. Scaling & precision

* **Model size** is set entirely by `KimiLinearConfig`. To go bigger than `small`,
  scale up `d_model`, `n_layers`, `moe_n_routed`, and `vocab_size`.
* **Sequence length:** `train_seq_len` **must be a multiple of `gdn_chunk_size`**
  (e.g. `tiny` uses 288 with chunk 32; `small` uses 256 with chunk 64).
* **Precision:** the model deliberately forces **fp32** in its numerically sensitive
  paths (log-decay, RMSNorm, MoE aux). `matmul_precision="bfloat16"` only sets JAX's
  matmul *accumulation* lever — a safe single-GPU speed-up; master weights stay fp32.
  For full bf16 *activations*, thread a `dtype=` into the `nnx.Linear` constructors in
  the model files.
* **Effective batch size** = `batch_size × grad_accum`. Increase `grad_accum` to keep a
  large effective batch when GPU memory limits `batch_size`.

---

## 13. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `pass@k` is always 0 with status `disabled` | `CODEGEN_ALLOW_EXEC` isn't set to `1`. Export it (see [§5 step 3](#step-3--evaluate-perplexity--functional-passk)). |
| `pass@k ≈ 0` on the `tiny` preset | Expected — `tiny` is a plumbing model, not a capable one. Use `small` on a GPU for real results. |
| `effective batch ... > train set` error | `batch_size × grad_accum` exceeds the dataset size. Lower `--batch-size` / `--grad-accum`. |
| `train_seq_len` errors / shape mismatch | `train_seq_len` must be a multiple of `gdn_chunk_size`. Adjust one of them. |
| Tokenizer file not found at train time | Run **step 1** (`python -m codegen.tokenizer --preset ...`) first; `train` loads `tokenizer_path`. |
| Out of memory on GPU | Lower `batch_size` (raise `grad_accum` to compensate), shorten `train_seq_len`, or shrink the model. |
| Runs on GPU but you wanted CPU (or vice-versa) | Set `JAX_PLATFORMS=cpu` to force CPU; install the CUDA JAX build to use a GPU. |
| MBPP download fails | First run needs network for `datasets`. After one successful download it's cached and works offline. |
| Loading a checkpoint errors about structure | You must load with the **same preset/config** the model was trained with. |

---

## 14. Repository layout

```
kimi_linear_gdn2.py          # the top-level model (3:1 GDN-2:MLA hybrid, MoE FFN, streaming generate)
gated_deltanet_2/            # GDN-2 linear-attention core (chunkwise train + recurrent decode) + layer
multi_latent_attention/      # MLA full-attention (NoPE) + dispatched grouped-GEMM MoE FFN
codegen/                     # the training / evaluation cycle:
  config.py                  #   tiny/small presets, the TrainConfig dataclass
  tokenizer.py               #   byte-level BPE (no <unk>, indentation preserved)
  data.py                    #   MBPP -> instruction prompts + completion-masked batches
  losses.py                  #   masked next-token cross-entropy (+ MoE aux) and pass@k estimator
  sampling.py                #   batched temperature / top-k / top-p decoding via streaming step
  sandbox.py                 #   execute generated code vs unit tests (subprocess + timeout + rlimits)
  checkpointing.py           #   Orbax save/restore of NNX model state
  evaluate.py                #   held-out perplexity + functional pass@k
  train.py                   #   the optimization loop (AdamW + warmup-cosine, grad-accum, MoE balancing)
main.ipynb                   # interactive end-to-end tutorial notebook
smoke_test.py                # end-to-end CPU verification (one script)
requirements.txt             # dependencies (install the JAX build for your accelerator)
README.md                    # architecture & design rationale
usage.md                     # this file
```

---

**Next steps:** run [`smoke_test.py`](smoke_test.py) or [`main.ipynb`](main.ipynb) to
see the cycle end-to-end, then move to the `small` preset on a GPU for a real run.
For the architectural reasoning, read [`README.md`](README.md).
