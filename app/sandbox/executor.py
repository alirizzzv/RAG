"""Sandboxed Python executor (Bullet 2: sandboxed execution framework).

Two backends, selected via settings.sandbox_backend:
  - "docker":     run code in a throwaway container, no network, mem/CPU caps
  - "subprocess": resource-limited child process (portable fallback)

TODO(P4): implement run(code) -> ExecResult(stdout, stderr, ok, artifact_path)
with a hard wall-clock timeout (settings.sandbox_timeout_seconds).
"""
from dataclasses import dataclass
from typing import Optional

from app.config import settings


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    artifact_path: Optional[str] = None


def run(code: str) -> ExecResult:
    raise NotImplementedError("Implemented in P4")
