"""
Autoregressive sampling for code generation.

Decoding reuses the model's streaming `step` / per-layer caches: prefill consumes
the prompt once, then each new token is O(1) work for the GDN-2 layers (fixed-size
recurrent state) and O(context) for the few MLA layers (growing latent cache).
The hot path — one streaming step + the logit filter + the categorical draw — is
fused under a single `nnx.jit` so the Python loop only carries token ids and keys.

Supported logit processors: temperature, top-k, and nucleus (top-p). `temperature
<= 0` switches to greedy argmax.

Batching convention: we sample `n_samples` continuations of ONE prompt at a time
(the whole batch shares the prompt, so there is no left-padding / per-row position
bookkeeping). The functional pass@k harness calls this once per problem.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from codegen.tokenizer import CodeTokenizer

NEG_INF = -1e30


def _filter_logits(logits: jax.Array, top_k: int, top_p: float) -> jax.Array:
    """Apply top-k then top-p masking to [B, V] logits (sets dropped logits -inf)."""
    B, V = logits.shape
    if top_k and 0 < top_k < V:
        kth = jax.lax.top_k(logits, top_k)[0][:, -1][:, None]  # k-th largest per row
        logits = jnp.where(logits < kth, NEG_INF, logits)
    if top_p and top_p < 1.0:
        order = jnp.argsort(logits, axis=-1)[:, ::-1]              # descending
        sorted_logits = jnp.take_along_axis(logits, order, axis=-1)
        probs = jax.nn.softmax(sorted_logits, axis=-1)
        cum = jnp.cumsum(probs, axis=-1)
        # Keep the smallest prefix whose cumulative prob first reaches top_p.
        keep_sorted = (cum - probs) < top_p
        keep_sorted = keep_sorted.at[:, 0].set(True)              # never drop the top token
        rows = jnp.arange(B)[:, None]
        keep = jnp.zeros_like(keep_sorted).at[rows, order].set(keep_sorted)
        logits = jnp.where(keep, logits, NEG_INF)
    return logits


@partial(nnx.jit, static_argnames=("temperature", "top_k", "top_p", "greedy"))
def _decode_step(model, ids, caches, key, *, temperature, top_k, top_p, greedy):
    """One streaming step -> (next_token [B,1], new_caches). Fused step+sample."""
    logits, caches = model.step(ids, caches)
    last = logits[:, -1, :].astype(jnp.float32)
    if greedy:
        nxt = jnp.argmax(last, axis=-1)
    else:
        last = _filter_logits(last / temperature, top_k, top_p)
        nxt = jax.random.categorical(key, last, axis=-1)
    return nxt[:, None].astype(ids.dtype), caches


def truncate_at_stops(text: str, stops: Sequence[str]) -> str:
    """Cut `text` at the earliest occurrence of any stop string."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def generate_batch(
    model: nnx.Module,
    prompt_ids: Sequence[int],
    n_samples: int,
    *,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 0.2,
    top_k: int = 0,
    top_p: float = 0.95,
    key: jax.Array | None = None,
) -> list[list[int]]:
    """Sample `n_samples` continuations of a single prompt.

    Returns a list of token-id lists (each already trimmed at the first EOS).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    greedy = temperature is not None and temperature <= 0.0

    B = n_samples
    P = len(prompt_ids)
    max_len = P + max_new_tokens
    prompt = jnp.asarray([prompt_ids] * B, dtype=jnp.int32)        # [B, P]

    caches = model.init_cache(B, max_len)
    # Prefill the whole prompt; first sampled token comes from its last position.
    key, sub = jax.random.split(key)
    nxt, caches = _decode_step(
        model, prompt, caches, sub,
        temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy,
    )

    collected = [nxt]                       # list of [B,1] arrays
    finished = jnp.asarray(nxt[:, 0] == eos_id)
    for _ in range(max_new_tokens - 1):
        if bool(jnp.all(finished)):         # host sync; cheap relative to a model step
            break
        key, sub = jax.random.split(key)
        nxt, caches = _decode_step(
            model, nxt, caches, sub,
            temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy,
        )
        finished = finished | (nxt[:, 0] == eos_id)
        collected.append(nxt)

    toks = jnp.concatenate(collected, axis=1)            # [B, T]
    toks_np = jax.device_get(toks)

    out: list[list[int]] = []
    for row in toks_np:
        seq = row.tolist()
        if eos_id in seq:                                # trim at first EOS
            seq = seq[: seq.index(eos_id)]
        out.append(seq)
    return out


def generate_completions(
    model: nnx.Module,
    tokenizer: CodeTokenizer,
    prompt: str,
    n_samples: int,
    *,
    stops: Sequence[str] = (),
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_k: int = 0,
    top_p: float = 0.95,
    key: jax.Array | None = None,
) -> list[str]:
    """Prompt-in / decoded-completions-out, with stop-string truncation applied."""
    prompt_ids = tokenizer.encode(prompt)
    seqs = generate_batch(
        model, prompt_ids, n_samples,
        max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id,
        temperature=temperature, top_k=top_k, top_p=top_p, key=key,
    )
    texts = [tokenizer.decode(s) for s in seqs]
    if stops:
        texts = [truncate_at_stops(t, stops) for t in texts]
    return texts
