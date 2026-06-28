"""
End-to-end smoke test of the whole code-generation cycle on the TINY (CPU) preset.

Runs, in one process:
  1. train a byte-level BPE tokenizer on MBPP,
  2. train the Kimi-Linear GDN-2 model for a handful of steps (loss must drop),
  3. evaluate held-out perplexity,
  4. sample completions, EXECUTE them against MBPP unit tests, score pass@k.

This is a *plumbing* test — the tiny model will not actually solve MBPP — but it
exercises every component the real `small` run uses. Expect a couple of minutes
on a laptop CPU.

    python smoke_test.py            # full cycle
    python smoke_test.py --steps 40 # shorter
"""

from __future__ import annotations

import argparse
import os
import warnings

# Keep CPU; make the cycle deterministic-ish and quiet.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CODEGEN_ALLOW_EXEC"] = "1"  # this is our own generated code on MBPP tests
warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120, help="training steps")
    ap.add_argument("--problems", type=int, default=5, help="MBPP test problems for pass@k")
    args = ap.parse_args()

    from codegen.config import get_preset
    from codegen.data import load_sft_datasets
    from codegen.evaluate import evaluate_pass_at_k, evaluate_perplexity
    from codegen.tokenizer import CodeTokenizer, _mbpp_corpus, train_tokenizer
    from codegen.train import train

    cfg = get_preset("tiny")
    cfg.max_steps = args.steps
    cfg.eval_every = max(args.steps // 2, 10)
    cfg.ckpt_every = args.steps
    cfg.eval_max_problems = args.problems

    print("=" * 70)
    print("STEP 1/4  train tokenizer")
    print("=" * 70)
    if not os.path.exists(cfg.tokenizer_path):
        train_tokenizer(_mbpp_corpus(cfg), cfg.vocab_size, save_path=cfg.tokenizer_path)
    tok = CodeTokenizer.load(cfg.tokenizer_path)
    print(f"  tokenizer vocab={tok.vocab_size}")

    print("\n" + "=" * 70)
    print(f"STEP 2/4  train model for {args.steps} steps")
    print("=" * 70)
    best_ckpt = train(cfg)

    print("\n" + "=" * 70)
    print("STEP 3/4  reload best checkpoint + perplexity")
    print("=" * 70)
    from codegen.checkpointing import load_checkpoint

    cfg.model.vocab_size = tok.vocab_size
    model = load_checkpoint(cfg.model, best_ckpt)
    _, val = load_sft_datasets(cfg, tok)
    ppl = evaluate_perplexity(model, val, cfg.batch_size)
    print(f"  val perplexity = {ppl:.2f}")

    print("\n" + "=" * 70)
    print(f"STEP 4/4  functional pass@k on {args.problems} MBPP problems")
    print("=" * 70)
    report = evaluate_pass_at_k(model, tok, cfg, verbose=False)
    for k, v in report["pass_at_k"].items():
        print(f"  pass@{k} = {v:.3f}")
    print(f"  exec status counts = {report['status_counts']}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — every stage ran end-to-end.")
    print("(A tiny 6-epoch CPU model is not expected to solve MBPP; pass@k≈0 is OK.)")
    print("Scale up with:  python -m codegen.train --preset small")
    print("=" * 70)


if __name__ == "__main__":
    main()
