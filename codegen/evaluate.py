"""
Evaluation: held-out perplexity + functional pass@k.

Perplexity measures how well the model predicts the *reference* solutions
(token-level, on the completion span only). It is cheap, needs no code execution,
and is the metric the training loop watches.

pass@k is the metric that actually matters for code generation: sample k programs
per task, execute them against the unit tests, and report the fraction of tasks
for which at least one sample passes (using the unbiased estimator over
`eval_n_samples >= max(ks)` draws). This requires running generated code, so it is
gated behind CODEGEN_ALLOW_EXEC=1 (see codegen/sandbox.py).

CLI:
    CODEGEN_ALLOW_EXEC=1 python -m codegen.evaluate --preset small \
        --ckpt runs/small/ckpt-best
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

warnings.filterwarnings("ignore", message=".*\\.value.*deprecated.*")

from codegen.config import TrainConfig, get_preset
from codegen.data import STOP_STRINGS, TokenizedDataset, load_eval_problems, load_sft_datasets
from codegen.losses import aggregate_pass_at_k, masked_next_token_loss
from codegen.sampling import generate_completions
from codegen.sandbox import count_correct, execution_enabled
from codegen.tokenizer import CodeTokenizer


# --------------------------------------------------------------------------- #
#  Perplexity
# --------------------------------------------------------------------------- #
@nnx.jit
def _eval_loss_step(model, ids, mask):
    logits, _ = model(ids)
    ce_mean, ntok = masked_next_token_loss(logits, ids, mask)
    return ce_mean * ntok, ntok  # summed CE, token count


def evaluate_perplexity(
    model: nnx.Module, ds: TokenizedDataset, batch_size: int = 16
) -> float:
    """Corpus perplexity over the completion tokens of `ds`."""
    if len(ds) == 0:
        return float("nan")
    total_ce, total_tok = 0.0, 0.0
    n = len(ds)
    for start in range(0, n, batch_size):
        ids = ds.input_ids[start : start + batch_size]
        mask = ds.loss_mask[start : start + batch_size]
        # Pad a ragged final batch up to batch_size with zero-mask rows.
        if ids.shape[0] < batch_size:
            pad = batch_size - ids.shape[0]
            ids = np.concatenate([ids, np.zeros((pad, ids.shape[1]), ids.dtype)], 0)
            mask = np.concatenate([mask, np.zeros((pad, mask.shape[1]), mask.dtype)], 0)
        ce_sum, ntok = _eval_loss_step(model, jnp.asarray(ids), jnp.asarray(mask))
        total_ce += float(ce_sum)
        total_tok += float(ntok)
    return float(np.exp(total_ce / max(total_tok, 1.0)))


# --------------------------------------------------------------------------- #
#  Functional pass@k
# --------------------------------------------------------------------------- #
def evaluate_pass_at_k(
    model: nnx.Module,
    tokenizer: CodeTokenizer,
    cfg: TrainConfig,
    *,
    split: str = "test",
    key: jax.Array | None = None,
    verbose: bool = True,
) -> dict:
    """Sample, execute, and score MBPP problems. Returns a report dict."""
    if key is None:
        key = jax.random.PRNGKey(cfg.seed)
    problems = load_eval_problems(cfg, split=split, max_problems=cfg.eval_max_problems)
    n_samples = max(cfg.eval_n_samples, max(cfg.eval_ks))

    if not execution_enabled():
        print(
            "[eval] CODEGEN_ALLOW_EXEC != 1 -> NOT executing generated code. "
            "pass@k will be reported as 0; set the env var to run the unit tests."
        )

    results: list[tuple[int, int]] = []
    status_counts: Counter = Counter()
    samples_log = []
    for i, prob in enumerate(problems):
        key, sub = jax.random.split(key)
        completions = generate_completions(
            model, tokenizer, prob.prompt, n_samples,
            stops=STOP_STRINGS,
            max_new_tokens=cfg.eval_max_new_tokens,
            temperature=cfg.eval_temperature,
            top_p=cfg.eval_top_p,
            key=sub,
        )
        c, exec_results = count_correct(
            completions, prob.test_list, prob.test_setup, cfg.eval_timeout
        )
        results.append((n_samples, c))
        for r in exec_results:
            status_counts[r.status] += 1
        if i == 0:
            samples_log.append((prob, completions, exec_results))
        if verbose and (i + 1) % 20 == 0:
            running = aggregate_pass_at_k(results, cfg.eval_ks)
            msg = "  ".join(f"pass@{k}={running[k]:.3f}" for k in cfg.eval_ks)
            print(f"[eval] {i + 1}/{len(problems)} problems  {msg}")

    passk = aggregate_pass_at_k(results, cfg.eval_ks)
    return {
        "n_problems": len(problems),
        "n_samples": n_samples,
        "pass_at_k": passk,
        "status_counts": dict(status_counts),
        "first_sample": samples_log[0] if samples_log else None,
    }


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained code-gen model.")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (from training)")
    ap.add_argument("--split", default="test", help="MBPP split for pass@k")
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-passk", action="store_true", help="perplexity only")
    args = ap.parse_args()

    from codegen.checkpointing import load_checkpoint

    cfg = get_preset(args.preset)
    if args.max_problems is not None:
        cfg.eval_max_problems = args.max_problems
    if args.n_samples is not None:
        cfg.eval_n_samples = args.n_samples
    if args.temperature is not None:
        cfg.eval_temperature = args.temperature

    tokenizer = CodeTokenizer.load(cfg.tokenizer_path)
    cfg.model.vocab_size = tokenizer.vocab_size
    print(f"[eval] loading checkpoint {args.ckpt}")
    model = load_checkpoint(cfg.model, args.ckpt)

    _, val = load_sft_datasets(cfg, tokenizer)
    ppl = evaluate_perplexity(model, val, cfg.batch_size)
    print(f"[eval] validation perplexity = {ppl:.3f}")

    if not args.no_passk:
        report = evaluate_pass_at_k(model, tokenizer, cfg, split=args.split)
        print(f"\n[eval] === MBPP {args.split} functional results ===")
        print(f"[eval] problems={report['n_problems']} samples/problem={report['n_samples']}")
        for k, v in report["pass_at_k"].items():
            print(f"[eval]   pass@{k} = {v:.4f}")
        print(f"[eval] exec status: {report['status_counts']}")
        # Show one generated sample for a qualitative look.
        fs = report["first_sample"]
        if fs is not None:
            prob, comps, execs = fs
            print(f"\n[eval] --- sample for task {prob.task_id} ---")
            print(prob.prompt + comps[0])
            print(f"[eval] result: {execs[0].status} ({execs[0].detail})")


if __name__ == "__main__":
    main()
