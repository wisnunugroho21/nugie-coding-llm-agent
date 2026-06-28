"""
Execute model-generated code against MBPP unit tests — the heart of *functional*
evaluation (does the code actually run and pass the asserts), as opposed to just
matching reference text.

SECURITY NOTE. Running model-generated code is inherently unsafe. This module is a
*pragmatic* sandbox, not a real one:
  * the program runs in a fresh child process (so it cannot corrupt the trainer),
  * with a hard wall-clock timeout (kills run-away / infinite-loop generations),
  * with POSIX resource limits (CPU seconds, address space) where available,
  * in a throwaway temp working directory.
It does NOT block network or filesystem access. For untrusted models at scale,
run this inside a container / gVisor / firejail. The harness honors the env var
CODEGEN_ALLOW_EXEC: set it to "1" to opt in. Without it, execution is refused.

The unit under test, assembled into one script:
    <test_setup>          # e.g. `class Pair: ...` or imports the asserts need
    <generated_code>      # the model's solution
    <assert 1> ... <assert n>     # MBPP test_list
A zero exit code (all asserts passed, no exception) == PASS.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Opt-in guard: code execution only runs when explicitly enabled.
EXEC_ENV_FLAG = "CODEGEN_ALLOW_EXEC"


def execution_enabled() -> bool:
    return os.environ.get(EXEC_ENV_FLAG, "") == "1"


@dataclass
class ExecResult:
    passed: bool
    status: str          # "passed" | "failed" | "timeout" | "error" | "disabled"
    detail: str = ""


def build_program(code: str, test_list, test_setup: str = "") -> str:
    """Assemble setup + solution + asserts into a single runnable script."""
    parts = []
    if test_setup.strip():
        parts.append(test_setup)
    parts.append(code)
    parts.extend(test_list)
    return "\n\n".join(parts) + "\n"


def _rlimit_preamble(cpu_seconds: int, mem_bytes: int) -> str:
    """Source prepended to the child program to cap CPU/memory FROM INSIDE the
    child. Doing this here (rather than via subprocess `preexec_fn`) avoids forcing
    the unsafe fork path in our multithreaded (JAX) parent process — the child can
    instead be launched with posix_spawn."""
    return (
        "import resource as _r\n"
        "try:\n"
        f"    _r.setrlimit(_r.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        f"    _r.setrlimit(_r.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))\n"
        "    _r.setrlimit(_r.RLIMIT_CORE, (0, 0))\n"
        "except Exception:\n"
        "    pass\n"
    )


def run_unit_test(
    code: str,
    test_list,
    test_setup: str = "",
    timeout: float = 8.0,
    mem_mb: int = 1024,
) -> ExecResult:
    """Run one (code, tests) pair in a subprocess. Returns an ExecResult."""
    if not execution_enabled():
        return ExecResult(False, "disabled", f"set {EXEC_ENV_FLAG}=1 to run tests")
    if not code.strip():
        return ExecResult(False, "error", "empty generation")

    program = build_program(code, test_list, test_setup)
    if os.name == "posix":
        program = _rlimit_preamble(
            cpu_seconds=max(1, int(timeout) + 1), mem_bytes=mem_mb * 1024 * 1024
        ) + program

    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", program],  # -I: isolated mode
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout", f">{timeout}s")
        except Exception as e:  # pragma: no cover - spawning failure
            return ExecResult(False, "error", repr(e)[:200])

    if proc.returncode == 0:
        return ExecResult(True, "passed")
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return ExecResult(False, "failed", detail[-1] if detail else f"rc={proc.returncode}")


def count_correct(
    solutions, test_list, test_setup: str = "", timeout: float = 8.0
) -> tuple[int, list[ExecResult]]:
    """Run a list of candidate solutions; return (#passed, per-solution results)."""
    results = [run_unit_test(s, test_list, test_setup, timeout) for s in solutions]
    return sum(r.passed for r in results), results
