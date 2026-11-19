"""Oracle: grader-side reference policy. Reads truth (handed over by the runner, never by
the environment) and submits the exact answer with a complete evidence set at step 1.
Used only to prove the maximum reward is reachable."""

from __future__ import annotations

from .base import Policy, submit


class OraclePolicy(Policy):
    name = "oracle"
    privileged = True

    def reset(self, obs: dict, truth: dict | None = None) -> None:
        super().reset(obs)
        if truth is None:
            raise ValueError("oracle needs truth")
        self.truth = truth

    def findings(self) -> list[dict]:
        return [{
            "type": t["type"],
            "affected_ids": list(t["affected_ids"]),
            "claimed_amount_minor": t["expected_amount_minor"],
            "currency": t["currency"],
            "evidence_ids": list(t["acceptable_evidence_sets"][0]),
            "explanation": "reference answer",
        } for t in self.truth["issues"]]

    def act(self, obs: dict) -> dict:
        return submit(self.episode_id, self.findings())
