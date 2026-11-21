"""judge_adapter.py - the organisers' uniform connection point (checks definitions, 2026-11-20).

Entry points (all JSON-compatible):

    generate_episode(seed, config, output_dir) -> None
    make_env() -> object with reset(seed, config) and step(action)
    score_episode(episode_dir, findings, inspection_steps) -> dict

This is a thin translation layer over ``audit_env``; it does not choose actions and does
not change the task.  Translation table (README "Judge adapter"):

    judge action  {"name": n, "args": a}      -> env action {"tool": n, "args": a}
    first obs     {"task", "manifest", "policy", "remaining_steps"}
    later obs     {"result": value|null, "error": null|{"code","message"}, "remaining_steps"}
    list_records  result {"records": [...], "next_cursor": str|null}
    get_record    result = the record object
    get_document  result = the document object (invoices include line_items)
    calculator    result = decimal string, e.g. "0.3"
    submit        result {"accepted": true}  or  error {"code": "INVALID_FINDINGS", ...}

Episode directory layout written by ``generate_episode`` (and read by reset/score):

    manifest.json  policy.json  ledger.csv  invoices.jsonl  payments.jsonl  fx.csv   (open)
    truth.json  provenance.json                                                     (closed)

Closed files may alternatively live in ``<episode_dir>/closed/``.  Only the grader
process (separate OS process spawned by the environment) and ``score_episode`` (judge
process) ever read truth.json / provenance.json; the environment object receives the
open bundle only and does not keep the directory path.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from audit_env import jsonutil  # noqa: E402
from audit_env.env.core import AuditEnv  # noqa: E402
from audit_env.env.tools import TOOL_HELP  # noqa: E402
from audit_env.generator.episode import generate_episode as _generate  # noqa: E402
from audit_env.generator.writer import read_closed, read_open, write_closed, write_open  # noqa: E402
from audit_env.grader.core import grade  # noqa: E402

_JUDGE_KEYS = ("mode", "episode_dir", "team_config")
GRADER_LOG = Path(os.environ.get("AUDIT_ENV_GRADER_LOG", str(_REPO / "runs" / "judge_grader_log.jsonl")))


def _team_config(config) -> dict | None:
    if not isinstance(config, dict):
        return None
    if isinstance(config.get("team_config"), dict):
        cfg = config["team_config"]
    else:
        cfg = {k: v for k, v in config.items() if k not in _JUDGE_KEYS}
    return dict(cfg) or None


# ----------------------------------------------------------------------------- generation
def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Generate one episode with the team generator and write it to ``output_dir``
    (open files + truth.json + provenance.json)."""
    ep = _generate(int(seed), _team_config(config))
    out = Path(output_dir)
    write_open(ep.bundle, out)
    write_closed(ep.truth, ep.provenance, out)


# ----------------------------------------------------------------------------- environment
def _translate_result(tool: str | None, result):
    if tool == "list_records":
        return {"records": result["records"], "next_cursor": result["next_cursor"]}
    if tool == "get_record":
        return result["record"]
    if tool == "get_document":
        return result["document"]
    if tool == "calculator":
        return result["value"]
    if tool == "submit":
        return {"accepted": True}
    return result


class JudgeEnv:
    """reset(seed, config) / step(action) in the organisers' JSON contract."""

    def __init__(self) -> None:
        self._envs: dict[str, AuditEnv] = {}
        self._env: AuditEnv | None = None

    def _get(self, mode: str) -> AuditEnv:
        if mode not in self._envs:
            self._envs[mode] = AuditEnv(mode=mode)
        return self._envs[mode]

    def reset(self, seed=None, config=None):
        config = config if isinstance(config, dict) else {}
        mode = "eval" if config.get("mode") in ("closed", "eval") else "train"
        self._env = self._get(mode)
        if config.get("episode_dir"):
            obs, info = self._env.reset(seed, {"episode_dir": str(config["episode_dir"])})
        else:
            obs, info = self._env.reset(seed, _team_config(config) or {})
        first = {
            "task": obs["task"] + " Tools (JSON action {\"name\": tool, \"args\": {...}}): "
                    + json.dumps(TOOL_HELP, sort_keys=True),
            "manifest": obs["manifest"],
            "policy": obs["policy"],
            "remaining_steps": obs["steps_remaining"],
        }
        return first, dict(info)

    def step(self, action):
        if self._env is None:
            raise RuntimeError("call reset() first")
        if isinstance(action, dict):
            inner = {"tool": action.get("name")}
            if "args" in action:
                inner["args"] = action["args"]
            for k in action:
                if k not in ("name", "args"):
                    inner[k] = action[k]  # unknown envelope keys -> BAD_ARGS, step consumed
        else:
            inner = action
        obs, reward, terminated, truncated, info = self._env.step(inner)
        tool = obs.get("tool")
        out = {
            "result": _translate_result(tool, obs["result"]) if obs.get("ok") else None,
            "error": None if obs.get("ok") else dict(obs["error"]),
            "remaining_steps": obs["steps_remaining"],
        }
        return out, float(reward), bool(terminated), bool(truncated), dict(info)

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()
        self._env = None

    def __del__(self) -> None:  # best effort
        try:
            self.close()
        except Exception:
            pass


def make_env() -> JudgeEnv:
    return JudgeEnv()


# ----------------------------------------------------------------------------- scoring
def _closed_dir(d: Path) -> Path:
    return d / "closed" if (d / "closed" / "truth.json").is_file() else d


def _log(entry: dict) -> None:
    try:
        GRADER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(GRADER_LOG, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(jsonutil.dumps(entry) + "\n")
    except OSError:
        pass


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    """Closed grader call (judge side, never a policy action)."""
    d = Path(episode_dir)
    bundle = read_open(d)
    truth, _ = read_closed(_closed_dir(d))
    steps = inspection_steps if isinstance(inspection_steps, int) and not isinstance(inspection_steps, bool) else 0
    steps = max(0, steps)
    res = grade(findings, truth, bundle, steps, "submitted")
    _log({"t": round(time.time(), 3), "episode_id": bundle["manifest"]["episode_id"],
          "inspection_steps": steps, "invalid_reason": res["invalid_reason"],
          "metrics": res["metrics"], "reward": res["reward"]})
    if res["invalid_reason"] is not None:
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None,
                "return": -1.0, "invalid_reason": res["invalid_reason"]}
    m = res["metrics"]
    return {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "return": res["reward"]}
