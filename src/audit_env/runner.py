"""Run policies against the environment, write trajectories, collect diagnostics.

The runner is harness code: for the oracle only, it loads truth itself (from
grader_data or by regenerating), never through the environment.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import __version__, jsonutil
from .env.core import AuditEnv
from .generator.episode import generate_episode
from .generator.writer import read_closed
from .grader.client import DEFAULT_GRADER_ROOT
from .policies.base import Policy


def privileged_truth(source: dict, grader_root: Path | None = None) -> dict:
    if "episode" in source:
        ep = source["episode"]
        truth, _ = read_closed((grader_root or DEFAULT_GRADER_ROOT) / ep["split"] / ep["episode_id"])
        return truth
    return generate_episode(source["seed"], source.get("config") or None).truth


def run_episode(env: AuditEnv, policy: Policy, seed: int | None = None, config: dict | None = None,
                grader_root: Path | None = None) -> tuple[list[dict], dict]:
    t0 = time.monotonic()
    obs, info = env.reset(seed=seed, config=config)
    truth = privileged_truth({"seed": seed, "config": config, **(config or {})}, grader_root) if policy.privileged else None
    policy.reset(obs, truth)
    common = {"policy": policy.name, "seed": seed, "version": __version__,
              "config_id": env.config_id or "", "mode": env.mode}
    rows = [dict(common, episode_id=env.episode_id, step=0, action=None, observation=obs,
                 reward=0.0, terminated=False, truncated=False)]
    terminated = truncated = False
    total = 0.0
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        rows.append(dict(common, episode_id=env.episode_id, step=obs["step"], action=jsonutil.deep_copy(action),
                         observation=obs, reward=reward, terminated=terminated, truncated=truncated))
    summary = {
        "episode_id": env.episode_id,
        "policy": policy.name,
        "seed": seed,
        "steps": rows[-1]["step"],
        "return": round(total, 6),
        "time_s": round(time.monotonic() - t0, 3),
        "outcome": "submitted" if terminated else "truncated",
    }
    if env.mode == "train":
        summary.update(reward=info["reward"], **info["metrics"], invalid_reason=info["invalid_reason"])
    return rows, summary


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(jsonutil.dumps(r) + "\n")


def format_table(summaries: list[dict]) -> str:
    by: dict[str, list[dict]] = {}
    for s in summaries:
        by.setdefault(s["policy"], []).append(s)
    cols = ["policy", "episodes", "P", "R", "F1", "reward", "steps", "time_s", "max_time_s"]
    lines = ["  ".join(f"{c:>10}" for c in cols)]
    for name in sorted(by):
        ss = by[name]
        n = len(ss)
        avg = lambda k: sum(s[k] for s in ss) / n  # noqa: E731
        vals = [name, n, avg("precision"), avg("recall"), avg("f1"), avg("reward"), avg("steps"),
                avg("time_s"), max(s["time_s"] for s in ss)]
        lines.append("  ".join(f"{v:>10.3f}" if isinstance(v, float) else f"{v:>10}" for v in vals))
    return "\n".join(lines)
