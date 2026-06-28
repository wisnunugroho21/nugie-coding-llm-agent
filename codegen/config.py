"""
Configuration for the code-generation training/eval cycle.

A `TrainConfig` bundles together:
  * `model`   - the `KimiLinearConfig` for the network (see kimi_linear_gdn2.py),
  * tokenizer / data / optimization / evaluation hyper-parameters.

Two presets are provided:

  * "tiny"  - deliberately small; the WHOLE cycle (tokenizer -> train -> pass@k)
              runs on a laptop CPU in a few minutes. Use it to verify the plumbing.
  * "small" - a ~200M-parameter (total) MoE model sized for a single GPU. Real (if
              modest) code-generation training; expect to actually need an accelerator.

Everything is plain dataclasses so a run is fully described (and serializable) by
one `TrainConfig`. CLI flags in train.py/evaluate.py override individual fields.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from kimi_linear_gdn2 import KimiLinearConfig

# Special tokens shared by the tokenizer, data pipeline and sampler.
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN]


@dataclass
class TrainConfig:
    name: str = "small"

    # --- model ---------------------------------------------------------------
    # vocab_size is overwritten at load time to match the trained tokenizer.
    model: KimiLinearConfig = field(default_factory=KimiLinearConfig)

    # --- tokenizer -----------------------------------------------------------
    vocab_size: int = 16000          # BPE target vocab (incl. special tokens)
    tokenizer_path: str = "runs/small/tokenizer.json"

    # --- data ----------------------------------------------------------------
    dataset: str = "mbpp"            # only "mbpp" is wired up here
    mbpp_config: str = "full"        # "full" (374 train) or "sanitized" (120 train)
    train_seq_len: int = 256         # padded length per example; MUST be a multiple
    #                                  of model.gdn_chunk_size (chunkwise core needs it)

    # --- optional Phase-1 pretraining (plain LM on a code corpus, before MBPP SFT) -
    # `pretrain_corpus` is either a LOCAL DIRECTORY of source files, or a HuggingFace
    # dataset id streamed via the `pretrain_hf_*` knobs. None -> SFT only.
    pretrain_corpus: str | None = None
    pretrain_hf_field: str = "content"     # text column for a HF dataset
    pretrain_hf_name: str | None = None    # HF config/name (e.g. "default")
    pretrain_hf_data_dir: str | None = None  # HF data_dir (e.g. "data/python")
    pretrain_max_docs: int = 5000          # cap #documents read (keeps it in RAM)
    pretrain_epochs: int = 1
    pretrain_max_steps: int | None = None  # overrides pretrain_epochs if set
    pretrain_lr: float | None = None       # defaults to `lr` if None

    # --- optimization --------------------------------------------------------
    batch_size: int = 16             # micro-batch size (sequences per forward)
    grad_accum: int = 2              # effective batch = batch_size * grad_accum
    epochs: int = 30                 # passes over the (small) MBPP train split
    max_steps: int | None = None     # if set, overrides epochs
    lr: float = 3e-4
    min_lr_ratio: float = 0.1        # final LR = lr * min_lr_ratio (cosine floor)
    warmup_ratio: float = 0.03       # warmup steps = warmup_ratio * total_steps
    weight_decay: float = 0.1        # applied to >=2-D kernels only (not norms/biases)
    grad_clip: float = 1.0
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    router_bias_lr: float = 1e-3     # aux-loss-free MoE balancing step (see moe.py)

    # --- precision -----------------------------------------------------------
    # The model forces fp32 in its numerically sensitive paths (decay, RMSNorm,
    # MoE aux). "bfloat16" here sets the matmul accumulation precision lever, a
    # safe single-GPU speed-up; master weights stay fp32. See README "Precision".
    matmul_precision: str = "highest"   # "highest" (fp32) | "high" | "bfloat16"

    # --- logging / checkpointing --------------------------------------------
    out_dir: str = "runs/small"
    log_every: int = 10
    eval_every: int = 200            # steps between perplexity evals
    passk_every: int = 0             # steps between (slow) pass@k evals; 0 = only at end
    ckpt_every: int = 200
    keep_ckpts: int = 3
    seed: int = 0

    # --- evaluation ----------------------------------------------------------
    eval_max_problems: int = 0       # 0 = all MBPP test problems
    eval_n_samples: int = 5          # samples per problem (for pass@k)
    eval_ks: tuple[int, ...] = (1, 5)
    eval_temperature: float = 0.2
    eval_top_p: float = 0.95
    eval_max_new_tokens: int = 256
    eval_timeout: float = 8.0        # seconds per unit-test execution

    def resolved_vocab(self) -> int:
        return self.model.vocab_size


# --------------------------------------------------------------------------- #
#  Presets
# --------------------------------------------------------------------------- #
def _tiny() -> TrainConfig:
    """CPU smoke-test scale: the full cycle runs in minutes, proves the plumbing."""
    model = KimiLinearConfig(
        vocab_size=2048,
        d_model=128,
        n_layers=4,
        full_attn_period=4,          # one MLA layer (index 3); rest GDN-2
        gdn_num_heads=2,
        gdn_head_k_dim=32,
        gdn_head_v_dim=32,
        gdn_chunk_size=32,
        mla_num_q_heads=4,
        mla_num_kv_heads=2,
        mla_head_dim=32,
        max_seq_len=320,
        moe_d_ff=128,
        moe_n_routed=4,
        moe_n_shared=1,
        moe_top_k=2,
        mlp_d_ff=256,
    )
    return TrainConfig(
        name="tiny",
        model=model,
        vocab_size=2048,
        tokenizer_path="runs/tiny/tokenizer.json",
        mbpp_config="sanitized",
        train_seq_len=288,           # 288 % 32 == 0; long enough for full solutions
        batch_size=8,
        grad_accum=1,
        epochs=8,
        lr=5e-4,
        warmup_ratio=0.05,
        out_dir="runs/tiny",
        log_every=2,
        eval_every=20,
        ckpt_every=40,
        eval_max_problems=10,
        eval_n_samples=3,
        eval_ks=(1, 3),
        eval_max_new_tokens=128,
    )


def _small() -> TrainConfig:
    """~200M-param (total) MoE model for a single GPU; far fewer active per token
    (top-2 of 8 experts), so compute/step is well under the parameter count."""
    model = KimiLinearConfig(
        vocab_size=16000,
        d_model=512,
        n_layers=12,
        full_attn_period=4,          # 3 GDN-2 : 1 MLA
        gdn_num_heads=8,
        gdn_head_k_dim=64,
        gdn_head_v_dim=64,
        gdn_chunk_size=64,
        mla_num_q_heads=8,
        mla_num_kv_heads=2,
        mla_head_dim=64,
        max_seq_len=512,
        moe_d_ff=1024,
        moe_n_routed=8,
        moe_n_shared=1,
        moe_top_k=2,
        mlp_d_ff=2048,
    )
    return TrainConfig(
        name="small",
        model=model,
        vocab_size=16000,
        tokenizer_path="runs/small/tokenizer.json",
        mbpp_config="full",
        train_seq_len=256,           # 256 % 64 == 0
        batch_size=16,
        grad_accum=2,
        epochs=40,
        lr=3e-4,
        out_dir="runs/small",
        matmul_precision="bfloat16",
        eval_every=200,
        ckpt_every=200,
        eval_n_samples=5,
        eval_ks=(1, 5),
    )


PRESETS = {"tiny": _tiny, "small": _small}


def get_preset(name: str) -> TrainConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return PRESETS[name]()


def as_dict(cfg: TrainConfig) -> dict:
    """Flat-ish dict for JSON logging (model config nested under 'model')."""
    d = dataclasses.asdict(cfg)
    return d
