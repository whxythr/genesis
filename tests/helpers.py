"""Shared test helpers."""

from audit_env.policies.base import submit


def oracle_findings(truth: dict) -> list[dict]:
    return [{
        "type": t["type"],
        "affected_ids": list(t["affected_ids"]),
        "claimed_amount_minor": t["expected_amount_minor"],
        "currency": t["currency"],
        "evidence_ids": list(t["acceptable_evidence_sets"][0]),
    } for t in truth["issues"]]


def submit_action(episode_id: str, findings: list[dict]) -> dict:
    return submit(episode_id, findings)
