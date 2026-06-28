"""
Data pipeline: MBPP -> instruction-formatted, completion-masked token batches.

MBPP ("Mostly Basic Python Problems") gives, per task: a natural-language
`description`, a reference `code` solution, and a `test_list` of `assert`
statements. The asserts reveal the expected function name/signature, so we put
them IN the prompt — exactly how the MBPP paper conditions the model.

We train SFT-style: each example becomes

    PROMPT       = "# Task: ...\n# tests...\n# Solution:\n"      (loss-masked OUT)
    COMPLETION   = "<reference code>\n<|endoftext|>"            (loss applies HERE)

so the model learns to *produce code given a task*, not to re-predict the prompt.
Sequences are right-padded to a fixed `seq_len` (a multiple of the GDN-2 chunk
size, which the chunkwise core requires). Padding and prompt tokens get loss_mask
0; completion tokens (incl. the terminating EOS) get loss_mask 1.

`iter_mbpp_raw` normalizes the two MBPP configs ("full" vs "sanitized", which use
different field names) into one schema. `load_eval_problems` returns the same
prompt plus the raw tests for the functional pass@k harness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from codegen.config import TrainConfig
from codegen.tokenizer import CodeTokenizer

# --------------------------------------------------------------------------- #
#  Prompt template
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = (
    "# Write a Python function to solve the task below.\n"
    "# Task: {description}\n"
    "# Your solution must pass these tests:\n"
    "{tests}\n"
    "# Solution:\n"
)

# Strings that, if generated, mean the model has run past its solution into the
# next (hallucinated) problem. Generation stops at the earliest of these or EOS.
STOP_STRINGS = ("\n# Write a Python function", "\n# Task:", "\n# Your solution")


def build_prompt(description: str, test_list: Sequence[str]) -> str:
    return PROMPT_TEMPLATE.format(
        description=description.strip(), tests="\n".join(test_list)
    )


def build_completion(code: str) -> str:
    return code.rstrip() + "\n"


# --------------------------------------------------------------------------- #
#  Raw MBPP loading / normalization
# --------------------------------------------------------------------------- #
def _normalize(ex: dict) -> dict:
    """Map either MBPP config's fields onto one schema."""
    description = ex.get("text") or ex.get("prompt") or ""
    setup = ex.get("test_setup_code", "") or ""
    if not setup and ex.get("test_imports"):
        setup = "\n".join(ex["test_imports"])
    return {
        "task_id": ex.get("task_id"),
        "description": description,
        "code": ex.get("code", ""),
        "test_list": list(ex.get("test_list", [])),
        "test_setup": setup,
    }


def iter_mbpp_raw(
    mbpp_config: str = "full", splits: Sequence[str] = ("train",)
) -> Iterator[dict]:
    """Yield normalized MBPP examples from the requested split(s)."""
    from datasets import load_dataset

    ds = load_dataset("mbpp", mbpp_config)
    for split in splits:
        if split not in ds:
            continue
        for ex in ds[split]:
            yield _normalize(ex)


# --------------------------------------------------------------------------- #
#  Tokenized SFT dataset
# --------------------------------------------------------------------------- #
@dataclass
class TokenizedDataset:
    input_ids: np.ndarray   # int32  [N, seq_len]
    loss_mask: np.ndarray   # uint8  [N, seq_len]  (1 = a completion target)

    def __len__(self) -> int:
        return self.input_ids.shape[0]


def build_sft_dataset(
    tokenizer: CodeTokenizer,
    examples: Sequence[dict],
    seq_len: int,
) -> TokenizedDataset:
    """Tokenize (prompt, completion) pairs into padded, loss-masked sequences."""
    pad_id = tokenizer.pad_id
    N = len(examples)
    ids_arr = np.full((N, seq_len), pad_id, dtype=np.int32)
    mask_arr = np.zeros((N, seq_len), dtype=np.uint8)

    kept = 0
    n_truncated = 0
    for ex in examples:
        prompt = build_prompt(ex["description"], ex["test_list"])
        completion = build_completion(ex["code"])
        p_ids = tokenizer.encode(prompt)
        c_ids = tokenizer.encode(completion, add_eos=True)

        ids = p_ids + c_ids
        mask = [0] * len(p_ids) + [1] * len(c_ids)
        if len(ids) > seq_len:
            ids, mask = ids[:seq_len], mask[:seq_len]
            n_truncated += 1
        # Skip examples whose completion got fully truncated away (no targets).
        if sum(mask) == 0:
            continue
        L = len(ids)
        ids_arr[kept, :L] = ids
        mask_arr[kept, :L] = mask
        kept += 1

    ds = TokenizedDataset(ids_arr[:kept], mask_arr[:kept])
    if n_truncated:
        print(f"[data] {n_truncated}/{N} examples truncated to seq_len={seq_len}")
    return ds


def load_sft_datasets(
    cfg: TrainConfig, tokenizer: CodeTokenizer
) -> tuple[TokenizedDataset, TokenizedDataset]:
    """Build (train, validation) tokenized datasets from MBPP."""
    train_raw = list(iter_mbpp_raw(cfg.mbpp_config, splits=("train",)))
    val_raw = list(iter_mbpp_raw(cfg.mbpp_config, splits=("validation",)))
    train = build_sft_dataset(tokenizer, train_raw, cfg.train_seq_len)
    val = build_sft_dataset(tokenizer, val_raw, cfg.train_seq_len)
    print(f"[data] SFT train={len(train)} val={len(val)} seq_len={cfg.train_seq_len}")
    return train, val


# --------------------------------------------------------------------------- #
#  Batch iteration  (yields the grad-accumulation axis up front)
# --------------------------------------------------------------------------- #
def batch_iterator(
    ds: TokenizedDataset,
    batch_size: int,
    grad_accum: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Iterator[dict]:
    """Yield effective batches shaped [grad_accum, batch_size, seq_len].

    One yield == one optimizer step. The leading axis is the micro-batch
    (gradient-accumulation) dimension consumed by train.py's train_step.
    """
    n = len(ds)
    eff = batch_size * grad_accum
    rng = np.random.default_rng(seed)
    order = rng.permutation(n) if shuffle else np.arange(n)
    limit = (n // eff) * eff if drop_last else n
    for start in range(0, limit, eff):
        idx = order[start : start + eff]
        if len(idx) < eff:  # only when drop_last=False and ragged tail
            pad = eff - len(idx)
            idx = np.concatenate([idx, order[:pad]])
        bi = ds.input_ids[idx].reshape(grad_accum, batch_size, -1)
        bm = ds.loss_mask[idx].reshape(grad_accum, batch_size, -1)
        yield {"input_ids": bi, "loss_mask": bm}


def steps_per_epoch(ds: TokenizedDataset, batch_size: int, grad_accum: int) -> int:
    return len(ds) // (batch_size * grad_accum)


# --------------------------------------------------------------------------- #
#  Evaluation problems (for functional pass@k)
# --------------------------------------------------------------------------- #
@dataclass
class EvalProblem:
    task_id: object
    prompt: str               # the text fed to the model
    test_list: list[str]      # assert statements
    test_setup: str           # setup code (class defs / imports) run before asserts
    reference_code: str       # gold solution (for reference / perplexity)


def load_eval_problems(
    cfg: TrainConfig, split: str = "test", max_problems: int = 0
) -> list[EvalProblem]:
    problems: list[EvalProblem] = []
    for ex in iter_mbpp_raw(cfg.mbpp_config, splits=(split,)):
        problems.append(
            EvalProblem(
                task_id=ex["task_id"],
                prompt=build_prompt(ex["description"], ex["test_list"]),
                test_list=ex["test_list"],
                test_setup=ex["test_setup"],
                reference_code=ex["code"],
            )
        )
        if max_problems and len(problems) >= max_problems:
            break
    return problems


# --------------------------------------------------------------------------- #
#  Optional: plain-LM packing for an extra code corpus (pretraining)
# --------------------------------------------------------------------------- #
def pack_lm_dataset(
    tokenizer: CodeTokenizer, texts: Sequence[str], seq_len: int
) -> TokenizedDataset:
    """Concatenate documents (EOS-separated) and chop into full seq_len blocks.
    Loss applies to every token (mask all ones). Used only if cfg.pretrain_corpus
    is set; the MBPP SFT path above is the default."""
    stream: list[int] = []
    for t in texts:
        stream.extend(tokenizer.encode(t, add_eos=True))
    n_blocks = len(stream) // seq_len
    if n_blocks == 0:
        return TokenizedDataset(
            np.zeros((0, seq_len), np.int32), np.zeros((0, seq_len), np.uint8)
        )
    arr = np.asarray(stream[: n_blocks * seq_len], np.int32).reshape(n_blocks, seq_len)
    return TokenizedDataset(arr, np.ones_like(arr, np.uint8))


def read_code_files(root: str, exts=(".py",)) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(exts):
                try:
                    with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                        out.append(fh.read())
                except (UnicodeDecodeError, OSError):
                    continue
    return out


def _iter_pretrain_texts(cfg: TrainConfig) -> Iterator[str]:
    """Yield document strings for Phase-1 pretraining, from either a local directory
    of source files or a streamed HuggingFace dataset (per cfg.pretrain_hf_*)."""
    src = cfg.pretrain_corpus
    assert src is not None
    if os.path.isdir(src):
        yield from read_code_files(src)
        return
    # Otherwise treat `src` as a HuggingFace dataset id and stream its text column.
    from datasets import load_dataset

    kwargs: dict = {"split": "train", "streaming": True}
    if cfg.pretrain_hf_name:
        kwargs["name"] = cfg.pretrain_hf_name
    if cfg.pretrain_hf_data_dir:
        kwargs["data_dir"] = cfg.pretrain_hf_data_dir
    ds = load_dataset(src, **kwargs)
    field = cfg.pretrain_hf_field
    for ex in ds:
        text = ex.get(field)
        if text:
            yield text


def load_pretrain_dataset(cfg: TrainConfig, tokenizer: CodeTokenizer) -> TokenizedDataset:
    """Build a packed, all-tokens-loss LM dataset from cfg.pretrain_corpus."""
    import itertools

    texts: Iterator[str] = _iter_pretrain_texts(cfg)
    if cfg.pretrain_max_docs:
        texts = itertools.islice(texts, cfg.pretrain_max_docs)
    texts = list(texts)
    print(f"[data] pretrain corpus: {len(texts)} documents from {cfg.pretrain_corpus!r}")
    ds = pack_lm_dataset(tokenizer, texts, cfg.train_seq_len)
    print(f"[data] pretrain packed into {len(ds)} blocks of seq_len={cfg.train_seq_len}")
    return ds
