"""Per-type failure breakdown of a reconcile policy (harness-side, uses truth after the run).

Usage: PYTHONPATH=src python scripts/failure_analysis.py [n_seeds] [--naive]
"""

import sys
from collections import Counter

from audit_env.env.core import AuditEnv
from audit_env.generator.episode import generate_episode
from audit_env.grader import matcher
from audit_env.grader.normalize import validate_and_normalize
from audit_env.policies.reconcile import ReconcilePolicy
from audit_env.runner import run_episode


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    naive = "--naive" in sys.argv
    env = AuditEnv()
    tp, fp, fn = Counter(), Counter(), Counter()
    rewards = []
    try:
        for seed in range(n):
            rows, summary = run_episode(env, ReconcilePolicy(naive_dates=naive), seed=seed)
            rewards.append(summary["reward"])
            ep = generate_episode(seed)
            findings, _ = validate_and_normalize(rows[-1]["action"]["args"]["findings"], ep.episode_id)
            pairs = matcher.match(findings, ep.truth["issues"], matcher.build_alias_map(ep.bundle))
            matched_f = {i for i, _ in pairs}
            matched_t = {t for _, t in pairs}
            for i, f in enumerate(findings):
                (tp if i in matched_f else fp)[f["type"]] += 1
            for t in ep.truth["issues"]:
                if t["issue_id"] not in matched_t:
                    fn[t["type"]] += 1
    finally:
        env.close()
    print(f"policy={'reconcile_naive' if naive else 'reconcile'} seeds={n} mean_reward={sum(rewards) / n:.3f}")
    print(f"{'type':20s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    for t in sorted(set(tp) | set(fp) | set(fn)):
        print(f"{t:20s} {tp[t]:5d} {fp[t]:5d} {fn[t]:5d}")


if __name__ == "__main__":
    main()
