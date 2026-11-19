"""Client side of the grader process (used by the environment and the runner)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .. import jsonutil
from ..config import REPO_ROOT

SRC_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EPISODES_ROOT = REPO_ROOT / "episodes"
DEFAULT_GRADER_ROOT = REPO_ROOT / "grader_data"


class GraderError(RuntimeError):
    pass


class GraderClient:
    def __init__(self, eval_mode: bool = False, episodes_root: Path | None = None,
                 grader_root: Path | None = None, log_path: Path | None = None) -> None:
        bootstrap = ("import sys; sys.path.insert(0, sys.argv[1]); "
                     "from audit_env.grader.service import main; main(sys.argv[2:])")
        cmd = [sys.executable, "-c", bootstrap, str(SRC_DIR),
               "--episodes-root", str(episodes_root or DEFAULT_EPISODES_ROOT),
               "--grader-root", str(grader_root or DEFAULT_GRADER_ROOT)]
        if eval_mode:
            cmd.append("--eval")
        if log_path:
            cmd += ["--log", str(log_path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def call(self, msg: dict) -> dict:
        if self._proc.poll() is not None:
            raise GraderError("grader process is not running")
        self._proc.stdin.write((jsonutil.dumps(msg) + "\n").encode("utf-8"))
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise GraderError("grader process died")
        return json.loads(line.decode("utf-8"))

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()

    def __del__(self) -> None:  # best effort
        try:
            self.close()
        except Exception:
            pass
