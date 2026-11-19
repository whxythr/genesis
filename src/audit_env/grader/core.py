"""Pure grading function: findings + truth + open bundle -> result."""

from __future__ import annotations

from . import matcher
from .metrics import prf
from .normalize import validate_and_normalize
from .reward import INVALID_REWARD, terminal_reward


def grade(raw_findings, truth: dict, bundle: dict, inspection_steps: int, outcome: str) -> dict:
    episode_id = bundle["manifest"]["episode_id"]
    issues = truth["issues"]
    base = {"episode_id": episode_id, "outcome": outcome, "inspection_steps": inspection_steps}
    if outcome == "truncated":
        findings, err = [], None
    else:
        findings, err = validate_and_normalize(raw_findings, episode_id)
    if err is not None:
        return dict(base, reward=INVALID_REWARD, invalid_reason=err,
                    metrics=prf(0, 0, len(issues)), matched_findings=[], n_findings=0)
    alias = matcher.build_alias_map(bundle)
    pairs = matcher.match(findings, issues, alias)
    tp = len(pairs)
    metrics = prf(tp, len(findings) - tp, len(issues) - tp)
    return dict(base, reward=terminal_reward(metrics["f1"], inspection_steps), invalid_reason=None,
                metrics=metrics, matched_findings=[i for i, _ in pairs], n_findings=len(findings))
