"""
Loss + metric helpers.

`masked_next_token_loss` is the standard causal-LM objective restricted to the
positions we care about (the completion tokens). Position t's logits predict
token t+1, so we shift by one and weight by the *target's* loss mask.

`pass_at_k` is the unbiased estimator from Chen et al. 2021 ("Evaluating Large
Language Models Trained on Code", the Codex/HumanEval paper): given n samples of
which c pass, the probability that at least one of a random size-k subset passes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

F32 = jnp.float32


def masked_next_token_loss(
    logits: jax.Array, input_ids: jax.Array, loss_mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Mean cross-entropy over masked completion targets.

    logits:    [B, L, V]   float
    input_ids: [B, L]      int    (the same ids that were fed in)
    loss_mask: [B, L]      0/1    (1 where a token is a completion target)

    Returns (loss, n_target_tokens). The loss is summed CE over targets divided
    by the number of targets (so it's per-token and batch-size invariant); the
    token count is returned so the caller can aggregate a corpus perplexity.
    """
    # Predict token t+1 from position t.
    logits = logits[:, :-1, :].astype(F32)
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:].astype(F32)

    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)  # [B, L-1]
    ce = ce * mask
    n_tok = mask.sum()
    loss = ce.sum() / jnp.maximum(n_tok, 1.0)
    return loss, n_tok


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-c, k) / C(n, k), computed stably (Chen et al. 2021).

    n = number of samples drawn, c = number that passed, k = the k in pass@k.
    """
    if k > n:
        raise ValueError(f"pass@{k} needs at least k={k} samples, got n={n}")
    if n - c < k:
        return 1.0
    # 1 - prod_{i=n-c+1..n} (1 - k/i)
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def aggregate_pass_at_k(
    results: list[tuple[int, int]], ks: tuple[int, ...]
) -> dict[int, float]:
    """results: list of (n_samples, n_correct) per problem -> {k: mean pass@k}."""
    out: dict[int, float] = {}
    for k in ks:
        vals = [pass_at_k(n, c, k) for (n, c) in results if n >= k]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
