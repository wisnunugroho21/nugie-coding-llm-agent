"""
Byte-level BPE tokenizer for code, built on HuggingFace `tokenizers`.

Why byte-level BPE for code generation:
  * Byte-level means there is NO out-of-vocabulary token — any byte (UTF-8 code,
    emoji in a comment, exotic identifier) is representable, so the model never
    hits an <unk>. The initial alphabet is the 256 byte representations.
  * BPE merges then recover the common code subwords (`def `, `return`, `    `
    indentation runs, `self.`, `__init__`), so a sequence of Python is ~3-4x
    shorter than its raw byte sequence — directly fewer tokens to train/generate.

Two special tokens only, kept minimal:
  * <|pad|>        - right-padding filler (loss-masked, never generated),
  * <|endoftext|>  - document / generation terminator (BOS is implicit).

`CodeTokenizer` wraps a trained `tokenizers.Tokenizer` and exposes the ids the
rest of the pipeline needs (pad_id, eos_id) plus batch encode/decode helpers.

CLI:  python -m codegen.tokenizer --preset small      # trains + saves the tokenizer
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Iterator, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from codegen.config import EOS_TOKEN, PAD_TOKEN, SPECIAL_TOKENS, TrainConfig, get_preset


class CodeTokenizer:
    """Thin wrapper around a trained byte-level BPE `tokenizers.Tokenizer`."""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.eos_id = tokenizer.token_to_id(EOS_TOKEN)
        if self.pad_id is None or self.eos_id is None:
            raise ValueError(
                "tokenizer is missing required special tokens "
                f"{SPECIAL_TOKENS!r}; retrain it with codegen.tokenizer"
            )

    # --- size ----------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    # --- encode / decode -----------------------------------------------------
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: Sequence[str], add_eos: bool = False) -> list[list[int]]:
        encs = self._tok.encode_batch(list(texts), add_special_tokens=False)
        out = [e.ids for e in encs]
        if add_eos:
            out = [ids + [self.eos_id] for ids in out]
        return out

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    # --- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "CodeTokenizer":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no tokenizer at {path!r}; train one first with "
                "`python -m codegen.tokenizer --preset <name>`"
            )
        return cls(Tokenizer.from_file(path))


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def _new_bpe() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    # Byte-level pre-tokenizer + decoder: lossless round-trip on arbitrary bytes,
    # and whitespace (crucial for code indentation) is preserved as visible tokens.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    corpus: Iterable[str],
    vocab_size: int,
    save_path: str | None = None,
    min_frequency: int = 2,
) -> CodeTokenizer:
    """Train a byte-level BPE on `corpus` (an iterable of strings)."""
    tok = _new_bpe()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,            # reserved ids 0,1 (pad, eos)
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    ct = CodeTokenizer(tok)
    if save_path:
        ct.save(save_path)
    return ct


def _mbpp_corpus(cfg: TrainConfig) -> Iterator[str]:
    """Yield raw text used to fit the tokenizer: every MBPP description, every
    reference solution, and the asserts — from the TRAIN/VALIDATION splits only
    (never the held-out test split, to avoid leaking test code into the vocab)."""
    from codegen.data import iter_mbpp_raw

    for ex in iter_mbpp_raw(cfg.mbpp_config, splits=("train", "validation", "prompt")):
        yield ex["description"]
        yield ex["code"]
        yield "\n".join(ex["test_list"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the byte-level BPE code tokenizer.")
    ap.add_argument("--preset", default="small", help="config preset (tiny|small)")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--out", default=None, help="override tokenizer save path")
    args = ap.parse_args()

    cfg = get_preset(args.preset)
    vocab = args.vocab_size or cfg.vocab_size
    out = args.out or cfg.tokenizer_path

    print(f"[tokenizer] training byte-level BPE: vocab={vocab} from MBPP/{cfg.mbpp_config}")
    ct = train_tokenizer(_mbpp_corpus(cfg), vocab_size=vocab, save_path=out)
    print(
        f"[tokenizer] saved -> {out}  (vocab_size={ct.vocab_size}, "
        f"pad_id={ct.pad_id}, eos_id={ct.eos_id})"
    )
    # quick sanity round-trip
    sample = "def add(a, b):\n    return a + b\n"
    ids = ct.encode(sample)
    print(f"[tokenizer] round-trip ok: {ct.decode(ids) == sample}  ({len(ids)} tokens)")


if __name__ == "__main__":
    main()
