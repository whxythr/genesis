"""AuditEnv: Gymnasium-style reset/step over one fictional company's books.

The environment process never holds truth: at ``reset`` it asks the grader process to
open (generate or load) an episode and receives only the open bundle. ``submit`` sends
the findings to the grader process, which returns the terminal reward (train mode) or
withholds it until the run is finalized (eval mode).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import __version__
from ..grader.client import GraderClient, GraderError
from .store import Store
from .tools import ERROR_MESSAGES, TOOL_HELP, TOOLS, ToolError, run

MAX_STEPS = 100
TIME_LIMIT_S = 60.0
MAX_ACTION_CHARS = 100_000

TASK = (
    "You are auditing one reporting period of a fictional company. Explore the books with the tools "
    "and submit findings for every real error. Error types: "
    "duplicate_invoice (affected_ids=[invoice_id of the later duplicate]; amount = its total in base currency), "
    "unapplied_payment (affected_ids=[payment_id]; amount = unapplied remainder in base currency; money applied "
    "to a duplicate invoice counts as unapplied), "
    "fx_error (affected_ids=[invoice_id or payment_id converted with a wrong rate]; amount = |posted - correct| in base currency), "
    "period_cutoff (affected_ids=[invoice_id or payment_id dated outside the period but posted in it]; "
    "amount = the operation converted at the policy rate, base currency). "
    "Amounts are integer minor units of the base currency at the policy rate of the document's own date, each "
    "computed as if the other errors were fixed; tolerance max(1, expected//1000). "
    "Each finding needs evidence_ids (doc_id/invoice_id/payment_id, txn_id/row_id, fx_id) that prove it. "
    "Correct but unusual activity (split or combined payments, refunds, subscriptions, late-evening or "
    "foreign-offset timestamps, FX conversions) is not an error. Clean books are possible: submit an empty "
    "list then. Every step costs a little reward; at most 100 steps including submit."
)


class AuditEnv:
    metadata = {"render_modes": []}

    def __init__(self, mode: str = "train", episodes_root: Path | None = None, grader_root: Path | None = None,
                 max_steps: int = MAX_STEPS, time_limit_s: float = TIME_LIMIT_S, grader_log: Path | None = None) -> None:
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s
        self._grader = GraderClient(eval_mode=(mode == "eval"), episodes_root=episodes_root,
                                    grader_root=grader_root, log_path=grader_log)
        self._store: Store | None = None
        self._steps = 0
        self._done = True
        self._terminated = False
        self._truncated = False
        self._t0 = 0.0
        self.episode_id: str | None = None
        self.config_id: str | None = None

    # ------------------------------------------------------------------------
    def reset(self, seed: int | None = None, config: dict | None = None):
        """Start an episode.

        ``config`` is either generator overrides (merged over configs/default.json) used
        with ``seed``, or ``{"episode": {"split": ..., "episode_id": ...}}`` to load a
        pre-generated episode from disk.
        """
        if self._store is not None and not self._done:
            self._grade(None, self._steps, "abandoned")
        config = dict(config or {})
        if "episode" in config:
            source = {"kind": "files", **config["episode"]}
        else:
            if seed is None:
                raise ValueError("seed is required to generate an episode")
            source = {"kind": "generate", "seed": seed, "config": config}
        reply = self._grader.call({"cmd": "open", "source": source})
        if not reply.get("ok"):
            raise GraderError(f"could not open episode: {reply.get('error')}")
        bundle = reply["bundle"]
        self._store = Store(bundle)
        self.episode_id = bundle["manifest"]["episode_id"]
        self.config_id = reply["config_id"]
        self._steps = 0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._t0 = time.monotonic()
        obs = {
            "task": TASK,
            "manifest": self._store.manifest,
            "policy": self._store.policy,
            "tools": TOOL_HELP,
            "steps_remaining": self.max_steps,
        }
        info = {} if self.mode == "eval" else {"episode_id": self.episode_id, "config_id": self.config_id}
        return obs, info

    # ------------------------------------------------------------------------
    def _obs(self, tool, ok: bool, result=None, error: str | None = None) -> dict:
        obs = {"step": self._steps, "steps_remaining": self.max_steps - self._steps, "tool": tool, "ok": ok}
        if ok:
            obs["result"] = result
        else:
            obs["error"] = {"code": error, "message": ERROR_MESSAGES[error]}
        return obs

    def _grade(self, findings, inspection_steps: int, outcome: str) -> tuple[float, dict]:
        reply = self._grader.call({"cmd": "grade", "findings": findings,
                                   "inspection_steps": inspection_steps, "outcome": outcome})
        self._done = True
        if not reply.get("ok"):
            raise GraderError(f"grading failed: {reply.get('error')}")
        if self.mode == "eval":
            return 0.0, {}
        res = reply["result"]
        info = {
            "reward": res["reward"],
            "metrics": res["metrics"],
            "outcome": res["outcome"],
            "inspection_steps": res["inspection_steps"],
            "matched_findings": res["matched_findings"],
            "invalid_reason": res["invalid_reason"],
        }
        return float(res["reward"]), info

    def step(self, action):
        if self._store is None:
            raise RuntimeError("call reset() first")
        if self._done:
            return self._obs(None, False, error="EPISODE_DONE"), 0.0, self._terminated, self._truncated, {}
        self._steps += 1
        tool = action.get("tool") if isinstance(action, dict) and isinstance(action.get("tool"), str) else None
        tool = tool if tool in TOOLS else None

        if time.monotonic() - self._t0 > self.time_limit_s:
            self._truncated = True
            reward, info = self._grade(None, self._steps, "truncated")
            return self._obs(tool, False, error="TIMEOUT"), reward, False, True, info

        err = self._validate_action(action)
        if err is None and action["tool"] == "submit":
            args = action.get("args")
            findings = args.get("findings") if isinstance(args, dict) else None
            self._terminated = True
            reward, info = self._grade(findings, self._steps - 1, "submitted")
            return self._obs("submit", True, {"submitted": True}), reward, True, False, info

        if err is None:
            try:
                obs = self._obs(tool, True, run(self._store, action["tool"], action.get("args", {})))
            except ToolError as e:
                obs = self._obs(tool, False, error=e.code)
        else:
            obs = self._obs(tool, False, error=err)

        if self._steps >= self.max_steps:
            self._truncated = True
            reward, info = self._grade(None, self._steps, "truncated")
            return obs, reward, False, True, info
        return obs, 0.0, False, False, {}

    @staticmethod
    def _validate_action(action) -> str | None:
        if not isinstance(action, dict):
            return "BAD_ARGS"
        try:
            if len(json.dumps(action)) > MAX_ACTION_CHARS:
                return "BAD_ARGS"
        except (TypeError, ValueError):
            return "BAD_ARGS"
        if set(action) - {"tool", "args"} or "tool" not in action:
            return "BAD_ARGS"
        if action["tool"] not in TOOLS:
            return "UNKNOWN_ACTION"
        if "args" in action and not isinstance(action["args"], dict):
            return "BAD_ARGS" if action["tool"] != "submit" else None
        return None

    def close(self) -> None:
        self._grader.close()

    # eval-mode helpers for the harness (never used by policies)
    def finalize(self) -> list[dict]:
        if self._store is not None and not self._done:
            self._grade(None, self._steps, "abandoned")
        self._grader.call({"cmd": "finalize"})
        reply = self._grader.call({"cmd": "report"})
        return reply.get("results", [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
