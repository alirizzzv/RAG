"""Sandboxed Python executor (Bullet 2: sandboxed execution framework).

Runs untrusted, LLM-generated code and returns a structured result. Two backends
(chosen via settings.sandbox_backend):

  - "subprocess": a child process in a throwaway temp dir, hard wall-clock
    timeout, CPU-time rlimit guard, minimal env. Portable — works anywhere
    (incl. Hugging Face Spaces). Default.
  - "docker": a throwaway container with --network none plus memory/CPU caps —
    stronger isolation for a real deployment. Needs a Docker daemon and an image
    that has pandas/matplotlib (configurable via SANDBOX_DOCKER_IMAGE).

Convention: generated code saves any chart/artifact to the file named
`output.png` in its working directory; the executor lifts it into artifact_dir.
"""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings

ARTIFACT_NAME = "output.png"


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    artifact_path: Optional[str] = None


def run(code: str) -> ExecResult:
    """Execute code in the configured sandbox and capture the outcome."""
    if settings.sandbox_backend == "docker":
        return _run_docker(code)
    return _run_subprocess(code)


def _collect_artifact(workdir: Path) -> Optional[str]:
    """Move a produced output.png out of the sandbox into artifact_dir."""
    produced = workdir / ARTIFACT_NAME
    if not produced.exists():
        return None
    dest_dir = Path(settings.artifact_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}.png"
    shutil.move(str(produced), dest)
    return str(dest)


# ----------------------------- subprocess backend -----------------------------

def _cpu_limit(seconds: int):
    """preexec hook: cap CPU seconds so a busy loop can't run forever."""
    def _apply():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        except (ValueError, OSError):
            pass
    return _apply


def _run_subprocess(code: str) -> ExecResult:
    timeout = settings.sandbox_timeout_seconds
    # Persistent matplotlib cache so the font list is built once, not per run
    # (a fresh HOME each time would otherwise rebuild it and blow the timeout).
    mpl_cache = Path(tempfile.gettempdir()) / "rag_sandbox_mplcache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "snippet.py").write_text(code)
        # Minimal environment; Agg backend so matplotlib needs no display.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(workdir),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(mpl_cache),
        }
        try:
            proc = subprocess.run(
                [sys.executable, "snippet.py"],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_cpu_limit(timeout) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "", f"TimeoutError: exceeded {timeout}s wall clock")

        artifact = _collect_artifact(workdir)
        return ExecResult(
            ok=(proc.returncode == 0),
            stdout=proc.stdout,
            stderr=proc.stderr,
            artifact_path=artifact,
        )


# ------------------------------- docker backend -------------------------------

def _run_docker(code: str) -> ExecResult:
    import docker  # imported lazily so subprocess mode needs no daemon

    image = os.environ.get("SANDBOX_DOCKER_IMAGE", "python:3.11-slim")
    timeout = settings.sandbox_timeout_seconds
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "snippet.py").write_text(code)
        client = docker.from_env()
        try:
            container = client.containers.run(
                image,
                command=["python", "/work/snippet.py"],
                volumes={str(workdir): {"bind": "/work", "mode": "rw"}},
                working_dir="/work",
                network_disabled=True,       # no network access
                mem_limit="512m",
                nano_cpus=1_000_000_000,      # 1 CPU
                environment={"MPLBACKEND": "Agg", "HOME": "/work"},
                detach=True,
            )
        except Exception as e:  # daemon down / image missing
            return ExecResult(False, "", f"DockerError: {e}")

        try:
            result = container.wait(timeout=timeout)
            logs = container.logs(stdout=True, stderr=False).decode(errors="replace")
            errs = container.logs(stdout=False, stderr=True).decode(errors="replace")
            ok = result.get("StatusCode", 1) == 0
        except Exception as e:
            container.kill()
            return ExecResult(False, "", f"TimeoutError/DockerError: {e}")
        finally:
            container.remove(force=True)

        artifact = _collect_artifact(workdir)
        return ExecResult(ok=ok, stdout=logs, stderr=errs, artifact_path=artifact)
