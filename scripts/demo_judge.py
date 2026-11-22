"""Walk through the jury contract step by step (for the demo video).

    python scripts/demo_judge.py [--pause 1.5]

Generates one episode with the team generator into a temp directory, then shows what the
jury sees through judge_adapter.py: closed mode (reward 0, empty info), bounded errors that
consume a step, the calculator, submit, and the true reward that only the grader computes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import judge_adapter as ja  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=1.5)
    args = ap.parse_args()

    def say(title: str) -> None:
        time.sleep(args.pause)
        print(f"\n=== {title}")

    def show(action: dict, out) -> None:
        obs, reward, term, trunc, info = out
        res = obs["result"]
        if isinstance(res, dict) and "records" in res:
            res = {"records": f"<{len(res['records'])} records>", "next_cursor": "<opaque>" if res["next_cursor"] else None}
        print(f"step  {json.dumps(action, ensure_ascii=False)}")
        print(f"  ->  result={json.dumps(res, ensure_ascii=False)[:110]}  error={obs['error']}")
        print(f"      reward={reward}  terminated={term}  truncated={trunc}  info={info}  "
              f"remaining_steps={obs['remaining_steps']}")

    matrix = json.loads((ROOT / "configs" / "judge_matrix.json").read_text(encoding="utf-8"))["scenarios"]
    ep = Path(tempfile.mkdtemp()) / "episode"
    say("1. generate_episode: interaction scenario 2 (duplicate invoice, partially paid)")
    ja.generate_episode(19, {"team_config": matrix["D19"]["team_config"]}, str(ep))
    print("files:", ", ".join(sorted(p.name for p in ep.iterdir())))

    env = ja.make_env()
    try:
        say("2. reset in CLOSED mode: the policy gets only task, manifest, policy, budget")
        obs, info = env.reset(None, {"mode": "closed", "episode_dir": str(ep), "team_config": {}})
        m = obs["manifest"]
        print("keys:", list(obs), " info:", info)
        print(f"episode {m['episode_id']}  base={m['base_currency']}  tz={m['timezone']}  "
              f"records={m['record_counts']}  remaining_steps={obs['remaining_steps']}")

        say("3. normal actions")
        for a in ({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}},
                  {"name": "calculator", "args": {"expression": "0.1+0.2"}}):
            show(a, env.step(a))

        say("4. bad input: bounded error, the step is consumed, nothing leaks")
        for a in ({"name": "get_record", "args": {"table": "ledger", "id": "NO-SUCH-ID"}},
                  {"name": "calculator", "args": {"expression": "open('truth.json').read()"}},
                  {"name": "rm -rf /", "args": {}}):
            show(a, env.step(a))

        say("5. submit correct findings (the oracle reads truth on the JURY side only)")
        truth = json.loads((ep / "truth.json").read_text(encoding="utf-8"))
        findings = {"episode_id": m["episode_id"], "findings": [
            {"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
             "currency": i["currency"], "evidence_ids": i["acceptable_evidence_sets"][0], "explanation": "demo"}
            for i in truth["issues"]]}
        for f in findings["findings"]:
            print(f"  {f['type']:<18} {f['affected_ids']}  {f['claimed_amount_minor']} {f['currency']}")
        show({"name": "submit", "args": {"findings": "<above>"}},
             env.step({"name": "submit", "args": {"findings": findings}}))
        print("closed mode: reward stays 0 for the policy, the true reward stays with the grader")
    finally:
        env.close()

    say("6. score_episode: the grader's verdict (5 inspection steps before submit)")
    print(json.dumps(ja.score_episode(str(ep), findings, 5)))
    say("7. attack: flag the same correct finding twice and add a false one")
    attack = dict(findings, findings=findings["findings"] * 2 + [dict(findings["findings"][0], claimed_amount_minor=1)])
    print(json.dumps(ja.score_episode(str(ep), attack, 5)))
    say("8. invalid structure (32 findings, limit is 30) -> -1")
    print(json.dumps(ja.score_episode(str(ep), dict(findings, findings=findings["findings"] * 16), 5)))


if __name__ == "__main__":
    main()
