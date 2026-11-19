"""`make demo`: a scripted policy audits one episode step by step in the terminal,
then the grader's P / R / F1 / reward and a per-finding verdict are printed."""

from __future__ import annotations

import time

from .env.core import AuditEnv
from .policies.reconcile import ReconcilePolicy

DEFAULT_EPISODE = ("train", "EP-88AR4KC9PB")  # has the partial-FX-payment-at-cutoff scenario


def _short(obs: dict, action: dict) -> str:
    tool = action.get("tool")
    args = action.get("args", {})
    if tool == "list_records":
        call = f"list_records({args.get('table')}{', next page' if args.get('cursor') else ''})"
    elif tool == "get_document":
        call = f"get_document({args.get('doc_id')})"
    elif tool == "get_record":
        call = f"get_record({args.get('table')}, {args.get('id')})"
    elif tool == "calculator":
        call = f"calculator({args.get('expression')!r})"
    else:
        call = f"submit({len(args.get('findings', {}).get('findings', []))} findings)"
    if not obs.get("ok"):
        out = f"ERROR {obs['error']['code']}"
    elif tool == "list_records":
        r = obs["result"]
        out = f"{len(r['records'])} of {r['total']} records" + ("" if r["next_cursor"] else " (last page)")
    elif tool == "get_document":
        d = obs["result"]["document"]
        if obs["result"]["doc_type"] == "invoice":
            out = f"invoice {d['invoice_id']} {d['total_minor']} {d['currency']} issued {d['issued_at']}"
        else:
            out = f"payment {d['payment_id']} {d['amount_minor']} {d['currency']} at {d['timestamp']}"
    elif tool == "calculator":
        out = obs["result"]["value"]
    else:
        out = "episode submitted"
    return f"{call:48s} -> {out}"


def run_demo(split: str = DEFAULT_EPISODE[0], episode_id: str = DEFAULT_EPISODE[1], delay: float = 0.0) -> int:
    env = AuditEnv(mode="train")
    try:
        obs, info = env.reset(config={"episode": {"split": split, "episode_id": episode_id}})
        m = obs["manifest"]
        print(f"Episode {m['episode_id']}  base={m['base_currency']}  tz={m['timezone']}")
        print(f"Period  {m['period_start']} .. {m['period_end']} (end exclusive)")
        print(f"Records ledger={m['record_counts']['ledger']} invoices={m['record_counts']['invoices']} "
              f"payments={m['record_counts']['payments']} fx={m['record_counts']['fx']}   budget={obs['steps_remaining']} steps\n")
        policy = ReconcilePolicy()
        policy.reset(obs)
        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, term, trunc, info = env.step(action)
            print(f"step {obs['step']:3d}  {_short(obs, action)}")
            if action["tool"] == "submit":
                findings = action["args"]["findings"]["findings"]
            done = term or trunc
            if delay:
                time.sleep(delay)
        matched = set(info["matched_findings"])
        print(f"\nSubmitted findings ({len(findings)}):")
        print(f"  {'':3s} {'type':18s} {'affected':14s} {'amount':>12s}  evidence")
        for i, f in enumerate(findings):
            mark = "TP" if i in matched else "FP"
            print(f"  {mark:3s} {f['type']:18s} {','.join(f['affected_ids']):14s} "
                  f"{f['claimed_amount_minor']:>12d}  {', '.join(f['evidence_ids'])}")
        mt = info["metrics"]
        print(f"\nGrader: TP={mt['tp']} FP={mt['fp']} FN={mt['fn']}  P={mt['precision']:.3f} "
              f"R={mt['recall']:.3f} F1={mt['f1']:.3f}  inspection_steps={info['inspection_steps']}  "
              f"reward={info['reward']:.3f}")
        print("reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)")
        return 0
    finally:
        env.close()
