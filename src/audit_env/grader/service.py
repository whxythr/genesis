"""Grader service: a separate OS process that owns truth.

Protocol: one JSON object per line on stdin -> one JSON object per line on stdout.

  {"cmd": "init", "episodes_root": str, "grader_root": str, "log": str}   (optional, once, first)

  {"cmd": "open", "source": {"kind": "generate", "seed": int, "config": {...}}}
  {"cmd": "open", "source": {"kind": "files", "split": str, "episode_id": str}}
      -> {"ok": true, "bundle": <open bundle>, "config_id": str}
  {"cmd": "grade", "findings": <raw>, "inspection_steps": int, "outcome": "submitted|truncated|abandoned"}
      -> {"ok": true, "result": {...}}   (train)   |   {"ok": true}   (eval: result withheld)
  {"cmd": "finalize"}  -> {"ok": true}         (eval: ends the session, no more opens)
  {"cmd": "report"}    -> {"ok": true, "results": [...]}   (eval: only after finalize)

Truth never leaves this process: no command returns truth, provenance or seed.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

from .. import jsonutil
from ..config import REPO_ROOT, ConfigError
from ..generator.episode import GenerationError, generate_episode
from ..generator.writer import read_closed, read_open
from .core import grade

_SPLIT_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_EP_RE = re.compile(r"^EP-[2-9A-HJ-NP-Z]{10}$")


class Service:
    def __init__(self, episodes_root: Path, grader_root: Path, eval_mode: bool, log_path: Path | None) -> None:
        self.episodes_root = episodes_root
        self.grader_root = grader_root
        self.eval_mode = eval_mode
        self.log_path = log_path
        self.active: dict | None = None
        self.results: list[dict] = []
        self.finalized = False
        self.initialized = False

    def log(self, entry: dict) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(jsonutil.dumps(entry) + "\n")

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd") if isinstance(msg, dict) else None
        if cmd == "init":
            if self.initialized:
                return {"ok": False, "error": "ALREADY_INITIALIZED"}
            self.initialized = True
            for key, attr in (("episodes_root", "episodes_root"), ("grader_root", "grader_root"), ("log", "log_path")):
                if isinstance(msg.get(key), str):
                    setattr(self, attr, Path(msg[key]))
            return {"ok": True}
        if cmd == "open":
            self.initialized = True
            return self._open(msg.get("source"))
        if cmd == "grade":
            return self._grade(msg)
        if cmd == "finalize":
            self.finalized = True
            self.active = None
            return {"ok": True}
        if cmd == "report":
            if self.eval_mode and not self.finalized:
                return {"ok": False, "error": "NOT_FINALIZED"}
            return {"ok": True, "results": self.results}
        if cmd == "ping":
            return {"ok": True}
        return {"ok": False, "error": "BAD_COMMAND"}

    def _open(self, source) -> dict:
        if self.finalized:
            return {"ok": False, "error": "FINALIZED"}
        if not isinstance(source, dict):
            return {"ok": False, "error": "BAD_SOURCE"}
        try:
            if source.get("kind") == "generate":
                seed = source.get("seed")
                if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                    return {"ok": False, "error": "BAD_SOURCE"}
                ep = generate_episode(seed, source.get("config") or None)
                bundle, truth, config_id = ep.bundle, ep.truth, ep.provenance["config_id"]
            elif source.get("kind") == "files":
                split, eid = source.get("split"), source.get("episode_id")
                if not (isinstance(split, str) and _SPLIT_RE.match(split) and isinstance(eid, str) and _EP_RE.match(eid)):
                    return {"ok": False, "error": "BAD_SOURCE"}
                open_dir = self.episodes_root / split / eid
                closed_dir = self.grader_root / split / eid
                if not open_dir.is_dir() or not closed_dir.is_dir():
                    return {"ok": False, "error": "NOT_FOUND"}
                bundle = read_open(open_dir)
                truth, provenance = read_closed(closed_dir)
                config_id = provenance["config_id"]
            else:
                return {"ok": False, "error": "BAD_SOURCE"}
        except (ConfigError, GenerationError, ValueError, KeyError, OSError):
            return {"ok": False, "error": "OPEN_FAILED"}
        self.active = {"bundle": bundle, "truth": truth, "config_id": config_id, "graded": False}
        return {"ok": True, "bundle": bundle, "config_id": config_id}

    def _grade(self, msg: dict) -> dict:
        if self.active is None or self.active["graded"]:
            return {"ok": False, "error": "NO_ACTIVE_EPISODE"}
        steps = msg.get("inspection_steps")
        outcome = msg.get("outcome")
        if not isinstance(steps, int) or outcome not in ("submitted", "truncated", "abandoned"):
            return {"ok": False, "error": "BAD_GRADE"}
        a = self.active
        result = grade(msg.get("findings"), a["truth"], a["bundle"], steps,
                       "truncated" if outcome == "abandoned" else outcome)
        result["outcome"] = outcome
        result["config_id"] = a["config_id"]
        a["graded"] = True
        self.results.append(result)
        self.log(result)
        if self.eval_mode:
            return {"ok": True}
        return {"ok": True, "result": result}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args(argv)
    svc = Service(REPO_ROOT / "episodes", REPO_ROOT / "grader_data", args.eval, None)
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            reply = {"ok": False, "error": "BAD_JSON"}
        else:
            reply = svc.handle(msg)
        stdout.write(jsonutil.dumps(reply) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
