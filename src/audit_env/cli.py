"""Command line: python -m audit_env {generate,validate,verify,baselines,transform,invariance,demo,check}."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__, jsonutil, schema
from .config import REPO_ROOT, config_id, resolve
from .env.core import AuditEnv
from .generator import evidence, invariants
from .generator.episode import Episode, generate_episode, write_episode
from .generator.writer import CLOSED_FILES, OPEN_FILES, read_closed, read_open
from .policies.flag_all import FlagAllPolicy
from .policies.oracle import OraclePolicy
from .policies.random_explorer import RandomExplorerPolicy
from .policies.reconcile import ReconcilePolicy
from .runner import format_table, run_episode, write_jsonl
from .transform import transform

EPISODES = REPO_ROOT / "episodes"
GRADER = REPO_ROOT / "grader_data"
TRAJ = REPO_ROOT / "trajectories"
_EP_DIR = re.compile(r"^EP-[2-9A-HJ-NP-Z]{10}$")


def policies():
    return [OraclePolicy(), ReconcilePolicy(), RandomExplorerPolicy(seed=0), FlagAllPolicy()]


def split_config(split: str) -> tuple[dict, list[int]]:
    with open(REPO_ROOT / "configs" / split / "v1.json", encoding="utf-8") as fh:
        spec = json.load(fh)
    return spec["config"], spec["seeds"]


def _episode_ids(root: Path, split: str) -> list[str]:
    d = root / split
    return sorted(p.name for p in d.iterdir() if p.is_dir() and _EP_DIR.match(p.name)) if d.is_dir() else []


def load_episode(split: str, eid: str, e_root: Path = EPISODES, g_root: Path = GRADER) -> Episode:
    truth, prov = read_closed(g_root / split / eid)
    return Episode(bundle=read_open(e_root / split / eid), truth=truth, provenance=prov)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _clear_split(split: str) -> None:
    for root in (EPISODES, GRADER):
        for eid in _episode_ids(root, split):
            shutil.rmtree(root / split / eid)


def write_split(split: str, e_root: Path, g_root: Path) -> list[Episode]:
    overrides, seeds = split_config(split)
    eps = [generate_episode(seed, overrides) for seed in seeds]
    for ep in eps:
        write_episode(ep, e_root, g_root, split)
    return eps


def cmd_generate(split: str) -> list[str]:
    overrides, seeds = split_config(split)
    _clear_split(split)
    eps = write_split(split, EPISODES, GRADER)
    cid = config_id(resolve(overrides))
    _write_text(EPISODES / split / "index.json",
                jsonutil.pretty({"split": split, "version": __version__, "config_id": cid,
                                 "episodes": sorted(e.episode_id for e in eps)}))
    # seeds and the episode composition live only on the grader side
    _write_text(GRADER / split / "seeds.json", jsonutil.pretty({
        "split": split, "seeds": seeds, "config": overrides,
        "composition": [{"episode_id": e.episode_id, "seed": e.provenance["seed"],
                         "issues": sorted(i["type"] for i in e.truth["issues"]),
                         "scenarios": e.provenance["generation"]["plan"]["scenarios"]} for e in eps]}))
    print(f"generated {len(eps)} {split} episodes ({cid})")
    return [e.episode_id for e in eps]


def cmd_validate(split: str) -> int:
    """Independent validation of the files on disk + byte-exact regeneration check."""
    failures = 0
    _, seeds = split_config(split)
    eids = _episode_ids(EPISODES, split)
    if len(eids) != len(seeds):
        print(f"[{split}] expected {len(seeds)} episodes, found {len(eids)}")
        failures += 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_split(split, tmp / "e", tmp / "g")
        for eid in eids:
            open_dir, closed_dir = EPISODES / split / eid, GRADER / split / eid
            ep = load_episode(split, eid)
            problems = [f"manifest: {e}" for e in schema.validate(ep.bundle["manifest"], "manifest")]
            problems += [f"truth: {e}" for e in schema.validate(ep.truth, "truth")]
            problems += invariants.compare_with_truth(ep.bundle, ep.truth)
            problems += evidence.check_sufficiency(ep.bundle, ep.truth)
            if sorted(p.name for p in open_dir.iterdir()) != sorted(OPEN_FILES):
                problems.append("unexpected files in the open directory")
            for name in OPEN_FILES:
                regen = tmp / "e" / split / eid / name
                if not regen.exists() or (open_dir / name).read_bytes() != regen.read_bytes():
                    problems.append(f"{name} differs from regeneration")
            for name in CLOSED_FILES:
                regen = tmp / "g" / split / eid / name
                if not regen.exists() or (closed_dir / name).read_bytes() != regen.read_bytes():
                    problems.append(f"{name} differs from regeneration")
            types = ",".join(sorted(i["type"] for i in ep.truth["issues"])) or "clean"
            scen = ",".join(ep.provenance["generation"]["plan"]["scenarios"])
            print(f"[{split}] {'OK  ' if not problems else 'FAIL'} {eid} ledger={ep.bundle['manifest']['record_counts']['ledger']:4d}"
                  f" docs={ep.bundle['manifest']['document_count']:3d} issues={len(ep.truth['issues']):2d} [{types}]"
                  + (f" scenarios=[{scen}]" if scen else ""))
            for p in problems:
                print(f"      - {p}")
            failures += bool(problems)
    return failures


def cmd_baselines(split: str, mode: str = "train", quiet: bool = False) -> list[dict]:
    """Run the 4 baselines. In eval mode each policy gets its own closed session and the
    rewards are revealed by the grader only after that session is finalized."""
    eids = _episode_ids(EPISODES, split)
    out_dir = TRAJ / (split if mode == "train" else f"{split}_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grader_log.jsonl").unlink(missing_ok=True)
    summaries: list[dict] = []
    all_rows: list[dict] = []
    for pol in policies():
        env = AuditEnv(mode=mode, grader_log=out_dir / "grader_log.jsonl")
        rows_all, pol_summaries = [], []
        try:
            for eid in eids:
                rows, summary = run_episode(env, pol, config={"episode": {"split": split, "episode_id": eid}})
                rows_all += rows
                pol_summaries.append(summary)
            if mode == "eval":
                results = env.finalize()
                if [r["episode_id"] for r in results] != [s["episode_id"] for s in pol_summaries]:
                    raise RuntimeError("grader report does not match the run")
                for s, r in zip(pol_summaries, results):
                    s.update(reward=r["reward"], invalid_reason=r["invalid_reason"], **r["metrics"])
        finally:
            env.close()
        all_rows += rows_all
        summaries += pol_summaries
    for stale in out_dir.glob("*.jsonl"):
        if stale.name not in ("trajectories.jsonl", "grader_log.jsonl"):
            stale.unlink()
    write_jsonl(all_rows, out_dir / "trajectories.jsonl")
    _write_text(out_dir / "summary.json", jsonutil.pretty([{k: v for k, v in s.items() if k != "time_s"} for s in summaries]))
    if not quiet:
        print(f"\nBaselines on {split}, mode={mode} ({len(eids)} episodes):")
        print(format_table(summaries))
    return summaries


def cmd_transform(split: str, seed: int, out_split: str) -> list[str]:
    """Write a permuted + renamed copy of a split (the ID mapping stays on the grader side)."""
    _clear_split(out_split)
    out = []
    for eid in _episode_ids(EPISODES, split):
        t, _ = transform(load_episode(split, eid), seed)
        write_episode(t, EPISODES, GRADER, out_split)
        out.append(t.episode_id)
    print(f"transformed {len(out)} {split} episodes -> {out_split} (seed={seed})")
    return out


def cmd_invariance(splits=("train", "dev"), seed: int = 7) -> int:
    """Paired checks: each episode vs. its permuted + renamed copy -> identical reward and F1."""
    failures = checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        eroot, groot = Path(tmp) / "e", Path(tmp) / "g"
        pairs = []
        for split in splits:
            for eid in _episode_ids(EPISODES, split):
                ep = load_episode(split, eid)
                t, _ = transform(ep, seed)
                write_episode(ep, eroot, groot, "orig")
                write_episode(t, eroot, groot, "trans")
                pairs.append((ep.episode_id, t.episode_id))
        env = AuditEnv(episodes_root=eroot, grader_root=groot)
        try:
            for a, b in pairs:
                for pol in (OraclePolicy(), ReconcilePolicy(), FlagAllPolicy()):
                    _, sa = run_episode(env, pol, config={"episode": {"split": "orig", "episode_id": a}}, grader_root=groot)
                    _, sb = run_episode(env, pol, config={"episode": {"split": "trans", "episode_id": b}}, grader_root=groot)
                    checks += 1
                    if (sa["reward"], sa["f1"], sa["steps"]) != (sb["reward"], sb["f1"], sb["steps"]):
                        failures += 1
                        print(f"  invariance FAIL {pol.name} {a}->{b}: {sa['reward']} vs {sb['reward']}")
        finally:
            env.close()
    print(f"paired order/ID checks: {checks} ({len(pairs)} episodes x 3 policies), failures: {failures}")
    return failures


def cmd_verify(paths: list[str] | None = None, split: str | None = None) -> int:
    """Independent file-level verification (audit_env.verify) of episode directories."""
    from .verify import verify_dir
    targets: list[tuple[Path, Path | None]] = [(Path(p), None) for p in paths or []]
    if split:
        targets += [(EPISODES / split / e, GRADER / split / e) for e in _episode_ids(EPISODES, split)]
    bad = 0
    for open_dir, closed_dir in targets:
        problems = verify_dir(open_dir, closed_dir)
        bad += bool(problems)
        print(f"[verify] {'OK  ' if not problems else 'FAIL'} {open_dir.name}"
              + "".join(f"\n         - {x}" for x in problems[:20]), flush=True)
    return bad


def cmd_check() -> int:
    steps = []

    def stage(name: str, ok: bool) -> None:
        steps.append((name, ok))
        print(f"==> {name}: {'OK' if ok else 'FAILED'}", flush=True)

    rc = subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=REPO_ROOT)
    stage("pytest", rc == 0)
    for split in ("train", "dev"):
        if not _episode_ids(EPISODES, split):
            cmd_generate(split)
        stage(f"validate {split} (invariants, truth, evidence, byte-exact regeneration)", cmd_validate(split) == 0)
        stage(f"verify {split} from files only (pre-injection rebuilt from provenance, declared "
              f"discrepancies, truth vs changes, evidence reachable, formats)", cmd_verify(split=split) == 0)
    ok = True
    for split, mode in (("train", "train"), ("dev", "eval")):
        summaries = cmd_baselines(split, mode)
        by: dict[str, list[dict]] = {}
        for s in summaries:
            by.setdefault(s["policy"], []).append(s)
        ok &= all(s["reward"] >= 0.99 and s.get("invalid_reason") is None for s in by["oracle"])
        ok &= sum(s["reward"] for s in by["flag_all"]) / len(by["flag_all"]) <= -0.5
        ok &= all(s["time_s"] <= 60 for s in summaries)
        mean = {k: sum(s["reward"] for s in v) / len(v) for k, v in by.items()}
        ok &= mean["reconcile"] > max(mean["random_explorer"], mean["flag_all"])
    stage("baselines train(open) + dev(closed eval): oracle ~ 1, reconcile > random/flag_all, "
          "flag_all low, <= 60 s/episode", ok)
    stage("invariance to row order and ID renaming (paired)", cmd_invariance() == 0)
    failed = [n for n, good in steps if not good]
    print("\nCHECK", "PASSED" if not failed else f"FAILED: {failed}")
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="audit_env")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "validate", "baselines"):
        p = sub.add_parser(name)
        p.add_argument("--split", default="train")
        if name == "baselines":
            p.add_argument("--mode", default="train", choices=["train", "eval"])
    p = sub.add_parser("transform")
    p.add_argument("--split", default="dev")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-split", default=None)
    sub.add_parser("invariance")
    p = sub.add_parser("verify", help="independent check of episode directories (Block 2 conditions)")
    p.add_argument("dirs", nargs="*")
    p.add_argument("--split", default=None)
    p = sub.add_parser("demo")
    p.add_argument("--split", default=None)
    p.add_argument("--episode", default=None)
    p.add_argument("--delay", type=float, default=0.0)
    sub.add_parser("check")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        cmd_generate(args.split)
        return 0
    if args.cmd == "validate":
        return 1 if cmd_validate(args.split) else 0
    if args.cmd == "baselines":
        cmd_baselines(args.split, args.mode)
        return 0
    if args.cmd == "transform":
        cmd_transform(args.split, args.seed, args.out_split or f"{args.split}_t{args.seed}")
        return 0
    if args.cmd == "invariance":
        return 1 if cmd_invariance() else 0
    if args.cmd == "verify":
        if not args.dirs and not args.split:
            ap.error("verify: give episode directories or --split")
        return 1 if cmd_verify(args.dirs, args.split) else 0
    if args.cmd == "demo":
        from .demo import DEFAULT_EPISODE, run_demo
        return run_demo(args.split or DEFAULT_EPISODE[0], args.episode or DEFAULT_EPISODE[1], args.delay)
    if args.cmd == "check":
        return cmd_check()
    return 2
