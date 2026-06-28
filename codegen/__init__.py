"""
codegen — a complete training and evaluation cycle for the Kimi-Linear (GDN-2)
language model, specialized for **programming code generation**.

The model itself lives in the repo root (`kimi_linear_gdn2.py`) and its building
blocks under `gated_deltanet_2/` and `multi_latent_attention/`. This package adds
everything *around* the model that a code-generation LLM needs:

    config        - tiny (CPU) / small (single-GPU) presets
    tokenizer     - byte-level BPE trained on the code corpus
    data          - MBPP -> instruction-formatted, completion-masked token batches
    losses        - masked next-token cross-entropy (+ MoE aux) and pass@k
    sampling      - temperature / top-k / top-p decoding via the streaming `step`
    sandbox       - execute generated code against unit tests (subprocess + timeout)
    checkpointing - orbax save / restore of nnx model state
    evaluate      - held-out perplexity + functional pass@k
    train         - the full optimization loop (warmup-cosine, grad-accum,
                    aux-loss-free MoE router-bias balancing, periodic eval + ckpt)

Typical end-to-end run (see README.md):

    python -m codegen.tokenizer  --preset small          # train the BPE tokenizer
    python -m codegen.train      --preset small          # train the model
    python -m codegen.evaluate   --preset small --ckpt runs/small/ckpt-best
"""

from codegen.config import PRESETS, TrainConfig, get_preset

__all__ = ["TrainConfig", "PRESETS", "get_preset"]
