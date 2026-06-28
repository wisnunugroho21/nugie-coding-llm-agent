# Kimi-Linear (GDN-2) — a code-generation LLM from scratch

A decoder-only language model for **programming code generation**, built in
JAX / Flax NNX, with a complete training + evaluation cycle on
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp).

The model follows the **Kimi Linear** architecture (a hybrid linear-attention
transformer) but substitutes **Gated DeltaNet-2** for Kimi Delta Attention as the
linear token mixer. Everything around the model — tokenizer, data, training loop,
sampler, and a *functional* `pass@k` evaluator that actually executes generated
code against unit tests — lives in the `codegen/` package.

---

## 1. Architecture

```
                       KimiLinear  (kimi_linear_gdn2.py)
  input_ids ─► Embed ─►┌───────────── DecoderLayer × n_layers ──────────────┐─► RMSNorm ─► LM head ─► logits
                       │  x += TokenMixer(RMSNorm(x))                        │
                       │       └ GDN-2 (linear)  on 3 of every 4 layers      │
                       │       └ MLA   (full attn) on every 4th layer (NoPE) │
                       │  x += MoE(RMSNorm(x))   # DeepSeek-style sparse FFN  │
                       └─────────────────────────────────────────────────────┘
```

| Component | File | Role |
|---|---|---|
| Top-level model | `kimi_linear_gdn2.py` | 3:1 GDN-2:MLA hybrid schedule, MoE FFN, streaming `generate` |
| GDN-2 token mixer | `gated_deltanet_2/` | linear attention with decoupled erase/write gates (chunkwise train + recurrent decode) |
| MLA full-attention | `multi_latent_attention/attention.py` | NoPE multi-head latent attention (the few non-linear layers) |
| MoE channel mixer | `multi_latent_attention/moe.py` | dispatched grouped-GEMM experts + aux-loss-free balancing |

**Why this is a good fit for code.** Source files are long and highly structured;
the linear GDN-2 layers give O(L) cost with a fixed-size recurrent state (cheap
long context), while the sparse 1:4 MLA layers restore the exact global lookups
that matter for matching brackets, variable references, and call signatures. The
MoE FFN adds capacity without proportional compute.

---

## 2. The `codegen/` cycle

| Module | What it does |
|---|---|
| `config.py` | `tiny` (CPU smoke) and `small` (~150M, single-GPU) presets |
| `tokenizer.py` | trains a byte-level BPE (no `<unk>`, indentation preserved) on the corpus |
| `data.py` | MBPP → instruction prompt + reference solution → **completion-masked** token batches |
| `losses.py` | masked next-token cross-entropy (+ MoE aux) and the unbiased `pass@k` estimator |
| `sampling.py` | batched temperature / top-k / top-p decoding via the streaming `step` |
| `sandbox.py` | runs generated code against unit tests (subprocess + timeout + rlimits) |
| `checkpointing.py` | Orbax save/restore of the NNX model state |
| `evaluate.py` | held-out perplexity + functional `pass@k` |
| `train.py` | the optimization loop (AdamW + warmup-cosine, grad-accum, MoE router-bias balancing, eval, ckpt) |

### Training objective
SFT-style next-token prediction. Each MBPP task becomes:

```
# Write a Python function to solve the task below.
# Task: <description>
# Your solution must pass these tests:
<assert ...>            ◄── the asserts reveal the function name/signature
# Solution:
<reference code><|endoftext|>     ◄── loss applies ONLY to this completion span
```

The loss mask zeroes out the prompt and padding, so the model learns *task → code*
rather than to re-predict the prompt.

### MoE load balancing
Two mechanisms, both already wired in:
* the small Switch/DeepSeek **softmax aux loss** returned by the model, added to the loss;
* the **aux-loss-free** per-expert selection bias (`update_router_bias`), nudged
  toward uniform load after every step *inside* the jitted train step (it mutates
  a non-trainable `nnx.Variable`, kept out of the gradient).

---

## 3. Quickstart

```bash
pip install -r requirements.txt          # install JAX matching your accelerator

# (a) verify the whole cycle end-to-end on CPU in ~2 minutes:
python smoke_test.py

# (b) a real single-GPU run on MBPP:
python -m codegen.tokenizer --preset small               # 1. train BPE (16k)
python -m codegen.train     --preset small               # 2. train the model
CODEGEN_ALLOW_EXEC=1 \
python -m codegen.evaluate  --preset small --ckpt runs/small/ckpt-best   # 3. pass@k
```

Outputs land in `runs/<preset>/`: `tokenizer.json`, `config.json`,
`train_log.jsonl`, `ckpt-best/`, `ckpt-last/`.

### Common overrides
```bash
python -m codegen.train --preset small --epochs 60 --lr 2e-4 --grad-accum 4
python -m codegen.evaluate --preset small --ckpt runs/small/ckpt-best \
       --max-problems 100 --n-samples 10 --temperature 0.4
```

---

## 4. Evaluation: what `pass@k` means here

For each MBPP test task we sample `n_samples` programs, execute each against the
task's `assert` tests, count how many pass (`c`), and report the unbiased
`pass@k = 1 - C(n-c, k)/C(n, k)` (Chen et al., 2021), averaged over tasks. We also
report **perplexity** on the held-out completion tokens (cheap, no execution).

> **Security.** `pass@k` runs model-generated code. It is gated behind
> `CODEGEN_ALLOW_EXEC=1` and uses a child process with a wall-clock timeout and
> POSIX CPU/memory limits — but it does **not** sandbox network/filesystem. For
> untrusted models run it inside a container / gVisor / firejail.

---

## 5. Scaling & precision

* **Model size** is set by `KimiLinearConfig` in `config.py`. The `small` preset is
  `d_model=512, n_layers=12, 8 experts (top-2)` ≈ 200M total params (far fewer
  active per token). Scale up `d_model`, `n_layers`, `moe_n_routed`, and
  `vocab_size` for larger runs.
* **Sequence length** (`train_seq_len`) must be a multiple of `gdn_chunk_size` —
  the GDN-2 chunkwise core reshapes `L` into `L/C` chunks.
* **Precision.** The model deliberately forces fp32 in its numerically sensitive
  paths (the log-decay, RMSNorm, MoE aux). `matmul_precision="bfloat16"` in the
  config sets JAX's matmul accumulation lever (a safe single-GPU speed-up; master
  weights stay fp32). For full bf16 activations, thread a `dtype=` into the
  `nnx.Linear` constructors in the model files.
* **More data (two-phase pretrain → SFT).** MBPP alone is small (374 train). For
  stronger models, pretrain on a Python corpus first (plain LM, all-token loss) and
  then SFT on MBPP — both phases run automatically when you pass `--pretrain-corpus`.
  The corpus is either a **local directory** of source files or a **streamed HF
  dataset**:

  ```bash
  # local directory of .py files
  python -m codegen.train --preset small \
      --pretrain-corpus /path/to/python/repo --pretrain-epochs 1

  # a HuggingFace code dataset (streamed; --pretrain-hf-* select the text column)
  python -m codegen.train --preset small \
      --pretrain-corpus codeparrot/codeparrot-clean-valid \
      --pretrain-hf-field content --pretrain-max-docs 20000
  ```

  Phase 1 packs the corpus into `train_seq_len` blocks and trains with its own
  warmup→cosine schedule (`--pretrain-lr`, `--pretrain-max-steps`), saves
  `ckpt-pretrain`, then Phase 2 continues those weights as MBPP SFT. Train the BPE
  tokenizer on the same corpus for best efficiency. The corpus is held in RAM
  (`--pretrain-max-docs` caps it); for very large corpora, stream-tokenize to a
  memmap (a small change in `data.pack_lm_dataset`).

---

## 6. Repo layout

```
kimi_linear_gdn2.py          # the model
gated_deltanet_2/            # GDN-2 linear-attention core + layer
multi_latent_attention/      # MLA attention + MoE FFN
codegen/                     # the training/evaluation cycle (this project)
  config.py  tokenizer.py  data.py  losses.py  sampling.py
  sandbox.py  checkpointing.py  evaluate.py  train.py
smoke_test.py                # end-to-end CPU verification
requirements.txt
```
