"""
Training loop for the Kimi-Linear (GDN-2) code-generation model.

What this loop does, beyond a vanilla LM trainer:

  * Objective = masked next-token cross-entropy on the *completion* span only
    (so the model learns task -> code), PLUS the MoE load-balancing aux loss the
    model already returns.

  * MoE balancing the Kimi/DeepSeek way: an *aux-loss-free* per-expert selection
    bias, nudged after every step toward uniform expert load via the model's own
    `update_router_bias`. This mutates a non-trainable `nnx.Variable` and is done
    INSIDE the jitted step (Flax NNX tracks the state mutation), kept out of the
    gradient. The small softmax aux loss is also added for extra balancing.

  * Gradient accumulation: the step takes a [grad_accum, batch, seq] tensor and
    averages grads over the micro-batches before a single optimizer update, so the
    effective batch can exceed device memory.

  * AdamW with warmup -> cosine decay, global-norm clipping, and weight decay on
    matmul kernels only (>=2-D params); RMSNorm weights, biases and the decay
    parameters are left undecayed.

  * Periodic held-out perplexity, best-checkpoint tracking, optional pass@k.

CLI:  python -m codegen.train --preset small
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

warnings.filterwarnings("ignore", message=".*\\.value.*deprecated.*")

from codegen.config import TrainConfig, as_dict, get_preset
from codegen.data import batch_iterator, load_sft_datasets, steps_per_epoch
from codegen.losses import masked_next_token_loss
from codegen.tokenizer import CodeTokenizer
from kimi_linear_gdn2 import KimiLinear, count_params
from multi_latent_attention.moe import update_router_bias


# --------------------------------------------------------------------------- #
#  Optimizer
# --------------------------------------------------------------------------- #
def _wd_mask(params):
    # Decay only >=2-D tensors (matmul kernels); leave norms/biases/decay params.
    return jax.tree.map(lambda p: jnp.ndim(p) >= 2, params)


def make_optimizer(model, cfg: TrainConfig, total_steps: int, lr: float | None = None):
    lr = cfg.lr if lr is None else lr
    warmup = max(1, int(cfg.warmup_ratio * total_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=max(total_steps, warmup + 1),
        end_value=lr * cfg.min_lr_ratio,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(
            schedule,
            b1=cfg.adam_b1,
            b2=cfg.adam_b2,
            weight_decay=cfg.weight_decay,
            mask=_wd_mask,
        ),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param), schedule


# --------------------------------------------------------------------------- #
#  Train step (grad accumulation + MoE router-bias update, all under one jit)
# --------------------------------------------------------------------------- #
def make_train_step(router_bias_lr: float):
    @nnx.jit
    def train_step(model, optimizer, input_ids, loss_mask):
        # input_ids, loss_mask: [A, B, L] (A = grad-accum micro-batches)
        A = input_ids.shape[0]

        def loss_fn(model, ids, mask):
            logits, aux = model(ids)
            ce, _ = masked_next_token_loss(logits, ids, mask)
            loss = ce + aux["aux_loss"]
            return loss, (ce, aux["aux_loss"], aux["group_sizes"])

        grads_sum = None
        ce_sum = jnp.float32(0.0)
        aux_sum = jnp.float32(0.0)
        gs_sum = None
        for a in range(A):
            (_, (ce, auxl, gs)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
                model, input_ids[a], loss_mask[a]
            )
            grads_sum = grads if grads_sum is None else jax.tree.map(jnp.add, grads_sum, grads)
            ce_sum += ce
            aux_sum += auxl
            gs_sum = gs if gs_sum is None else gs_sum + gs

        grads = jax.tree.map(lambda g: g / A, grads_sum)
        optimizer.update(model, grads)

        # Aux-loss-free MoE balancing: update each layer's selection bias outside
        # the gradient, using accumulated per-expert token counts.
        for i, layer in enumerate(model.layers):
            moe = layer.channel_mixer
            moe.router_bias.value = update_router_bias(
                moe.router_bias.value, gs_sum[i], router_bias_lr
            )

        return ce_sum / A, aux_sum / A

    return train_step


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig) -> str:
    if cfg.matmul_precision:
        jax.config.update("jax_default_matmul_precision", cfg.matmul_precision)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # --- tokenizer (train it first via codegen.tokenizer if missing) ---
    tokenizer = CodeTokenizer.load(cfg.tokenizer_path)
    cfg.model.vocab_size = tokenizer.vocab_size
    print(f"[train] tokenizer vocab={tokenizer.vocab_size} (pad={tokenizer.pad_id} eos={tokenizer.eos_id})")

    # --- model + shared train step ---
    rngs = nnx.Rngs(cfg.seed)
    model = KimiLinear(cfg.model, rngs=rngs)
    n_params = count_params(model)
    print(f"[train] model params={n_params:,} ({n_params / 1e6:.1f}M)")
    train_step = make_train_step(cfg.router_bias_lr)

    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({"n_params": n_params, **as_dict(cfg)}, f, indent=2, default=str)

    from codegen.checkpointing import save_checkpoint

    log_f = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "w")
    abs_step = 0

    # === Phase 1 (optional): plain-LM pretraining on a code corpus ===========
    # Same train_step — pack_lm_dataset emits an all-ones loss mask, so the masked
    # cross-entropy degenerates to standard next-token LM over every token.
    if cfg.pretrain_corpus:
        from codegen.data import load_pretrain_dataset

        pre_ds = load_pretrain_dataset(cfg, tokenizer)
        spe = steps_per_epoch(pre_ds, cfg.batch_size, cfg.grad_accum)
        if spe == 0:
            raise RuntimeError(
                f"pretrain corpus packs to {len(pre_ds)} blocks < one effective batch "
                f"({cfg.batch_size * cfg.grad_accum}); add more data or lower the batch."
            )
        pre_steps = cfg.pretrain_max_steps or (cfg.pretrain_epochs * spe)
        pre_lr = cfg.pretrain_lr or cfg.lr
        print(f"[train] === PRETRAIN: {len(pre_ds)} blocks  steps/epoch={spe}  "
              f"total={pre_steps}  lr={pre_lr:.2e} ===")
        opt, sched = make_optimizer(model, cfg, pre_steps, lr=pre_lr)
        abs_step = _train_loop(
            model, opt, train_step, sched, tokenizer, pre_ds, None, cfg,
            pre_steps, "pretrain", log_f, start_step=abs_step,
        )
        save_checkpoint(model, os.path.join(cfg.out_dir, "ckpt-pretrain"),
                        meta={"step": abs_step, "phase": "pretrain"})
        print(f"[train] pretrain done -> {os.path.join(cfg.out_dir, 'ckpt-pretrain')}")

    # === Phase 2: SFT on MBPP (completion-masked) ============================
    train_ds, val_ds = load_sft_datasets(cfg, tokenizer)
    spe = steps_per_epoch(train_ds, cfg.batch_size, cfg.grad_accum)
    if spe == 0:
        raise RuntimeError(
            f"effective batch {cfg.batch_size * cfg.grad_accum} > train set {len(train_ds)}; "
            "lower batch_size/grad_accum."
        )
    sft_steps = cfg.max_steps or (cfg.epochs * spe)
    print(f"[train] === SFT: steps/epoch={spe}  total_steps={sft_steps}  "
          f"epochs~={sft_steps / spe:.1f} ===")
    opt, sched = make_optimizer(model, cfg, sft_steps, lr=cfg.lr)
    best = {"ppl": float("inf"), "path": os.path.join(cfg.out_dir, "ckpt-best")}
    abs_step = _train_loop(
        model, opt, train_step, sched, tokenizer, train_ds, val_ds, cfg,
        sft_steps, "sft", log_f, start_step=abs_step, best=best,
    )

    save_checkpoint(model, os.path.join(cfg.out_dir, "ckpt-last"), meta={"step": abs_step})
    log_f.close()
    print(f"[train] done. best val_ppl={best['ppl']:.3f}  best ckpt={best['path']}")
    return best["path"]


def _train_loop(
    model, optimizer, train_step, schedule, tokenizer, train_ds, val_ds, cfg,
    total_steps, phase, log_f, *, start_step=0, best=None,
):
    """Run `total_steps` optimizer steps over `train_ds`. When `val_ds` is given,
    track best held-out perplexity into `best` (a {'ppl','path'} dict) and save the
    best checkpoint. Returns the absolute step reached (for multi-phase continuity)."""
    from codegen.checkpointing import save_checkpoint
    from codegen.evaluate import evaluate_perplexity

    step = start_step
    target = start_step + total_steps
    t0 = time.time()
    data_seed = cfg.seed
    while step < target:
        for batch in batch_iterator(train_ds, cfg.batch_size, cfg.grad_accum, seed=data_seed):
            ce, auxl = train_step(
                model, optimizer,
                jnp.asarray(batch["input_ids"]), jnp.asarray(batch["loss_mask"]),
            )
            step += 1
            local = step - start_step  # phase-local step (indexes this phase's schedule)

            if local % cfg.log_every == 0:
                ce_f = float(ce)
                lr = float(schedule(local))
                tput = local * cfg.batch_size * cfg.grad_accum / (time.time() - t0)
                rec = {"phase": phase, "step": step, "ce": ce_f, "ppl": float(jnp.exp(ce)),
                       "aux": float(auxl), "lr": lr, "seq_per_s": tput}
                print(f"[{phase}] step {step:>6} ({local}/{total_steps})  ce={ce_f:.4f}  "
                      f"ppl={rec['ppl']:.2f}  lr={lr:.2e}  {tput:.0f} seq/s")
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()

            if val_ds is not None and (local % cfg.eval_every == 0 or step == target):
                val_ppl = evaluate_perplexity(model, val_ds, cfg.batch_size)
                print(f"[{phase}] step {step} >>> val_ppl={val_ppl:.3f}  (best={best['ppl']:.3f})")
                log_f.write(json.dumps({"phase": phase, "step": step, "val_ppl": val_ppl}) + "\n")
                log_f.flush()
                if val_ppl < best["ppl"]:
                    best["ppl"] = val_ppl
                    save_checkpoint(model, best["path"], meta={"step": step, "val_ppl": val_ppl})
                    print(f"[{phase}] new best -> {best['path']}")

            if cfg.ckpt_every and local % cfg.ckpt_every == 0:
                save_checkpoint(model, os.path.join(cfg.out_dir, "ckpt-last"),
                                meta={"step": step, "phase": phase})

            if cfg.passk_every and val_ds is not None and local % cfg.passk_every == 0:
                _run_passk(model, tokenizer, cfg)

            if step >= target:
                break
        data_seed += 1  # reshuffle each epoch
    return step


def _run_passk(model, tokenizer, cfg: TrainConfig):
    from codegen.evaluate import evaluate_pass_at_k

    rep = evaluate_pass_at_k(model, tokenizer, cfg, verbose=False)
    line = "  ".join(f"pass@{k}={rep['pass_at_k'][k]:.3f}" for k in cfg.eval_ks)
    print(f"[train] pass@k ({rep['n_problems']} problems): {line}  status={rep['status_counts']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Kimi-Linear GDN-2 code model.")
    ap.add_argument("--preset", default="small", help="tiny|small")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    # Phase-1 pretraining on a code corpus (local dir or HF dataset id).
    ap.add_argument("--pretrain-corpus", default=None,
                    help="local dir of source files, or a HuggingFace dataset id")
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--pretrain-max-steps", type=int, default=None)
    ap.add_argument("--pretrain-max-docs", type=int, default=None)
    ap.add_argument("--pretrain-lr", type=float, default=None)
    ap.add_argument("--pretrain-hf-field", default=None, help="HF text column (default 'content')")
    ap.add_argument("--pretrain-hf-name", default=None, help="HF dataset config/name")
    ap.add_argument("--pretrain-hf-data-dir", default=None, help="HF data_dir (e.g. data/python)")
    args = ap.parse_args()

    cfg = get_preset(args.preset)
    overrides = (
        "epochs", "max_steps", "batch_size", "grad_accum", "lr", "out_dir", "seed",
        "pretrain_corpus", "pretrain_epochs", "pretrain_max_steps", "pretrain_max_docs",
        "pretrain_lr", "pretrain_hf_field", "pretrain_hf_name", "pretrain_hf_data_dir",
    )
    for attr in overrides:
        v = getattr(args, attr)
        if v is not None:
            setattr(cfg, attr, v)
    train(cfg)


if __name__ == "__main__":
    main()
