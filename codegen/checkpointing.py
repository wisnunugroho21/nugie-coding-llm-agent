"""
Checkpointing for the Flax NNX model, via Orbax.

We persist only the model *state* (the pytree of parameter + variable arrays from
`nnx.split`). To restore we rebuild the module structure abstractly with
`nnx.eval_shape` (no host/device allocation), restore the arrays into that
template, and `nnx.merge` graph-def + state back into a live module. This is the
pattern recommended in the Flax NNX docs, and it means a checkpoint is portable as
long as the `KimiLinearConfig` used to rebuild matches.

A tiny JSON sidecar records step / metric so we can pick the best checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from typing import Any

import flax.nnx as nnx
import orbax.checkpoint as ocp

# Single-device restore: Orbax warns that no sharding was provided. Benign here.
warnings.filterwarnings("ignore", message=".*Sharding info not provided.*")

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig


def _abs(path: str) -> str:
    return os.path.abspath(path)


def save_checkpoint(model: KimiLinear, path: str, meta: dict[str, Any] | None = None) -> None:
    """Save model state (overwriting any existing checkpoint at `path`)."""
    path = _abs(path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    _, state = nnx.split(model)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(os.path.join(path, "state"), state)
    ckptr.wait_until_finished()

    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta or {}, f, indent=2, default=str)


def load_checkpoint(cfg: KimiLinearConfig, path: str) -> KimiLinear:
    """Rebuild a KimiLinear from a checkpoint produced by `save_checkpoint`."""
    path = _abs(path)
    state_dir = os.path.join(path, "state")
    if not os.path.exists(state_dir):
        raise FileNotFoundError(f"no checkpoint state at {state_dir!r}")

    abstract = nnx.eval_shape(lambda: KimiLinear(cfg, rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract)

    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(state_dir, abstract_state)
    return nnx.merge(graphdef, restored)


def read_meta(path: str) -> dict[str, Any]:
    meta_path = os.path.join(_abs(path), "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)
