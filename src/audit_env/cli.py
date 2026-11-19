"""Command line: python -m audit_env {generate,validate,baselines,check}."""

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
from .generator import invariants
from .generator.episode import generate_episode, write_episode
from .generator.writer import CLOSED_FILES, OPEN_FILES, read_closed, read_open
from .policies.flag_all import FlagAllPolicy
from .policies.oracle import OraclePolicy
from .runner import format_table, run_episode, write_jsonl

EPISODES = REPO_ROOT / "episodes"
GRADER = REPO_ROOT / "grader_data"
TRAJ = REPO_ROOT / "trajectories"
_EP_DIR = re.compile(r"^EP-[2-9A-HJ-NP-Z]{10}$")


def policies():
    out = [OraclePolicy(), FlagAllPolicy()]
    try:
        from .policies.random_explorer import RandomExplorerPolicy
        from .policies.reconcile import ReconcilePolicy
        out += [RandomExplorerPolicy(seed=0), ReconcilePolicy()]
    except ImportError:
        pass
    return out


def split_config(split: str) -> tuple[dict, list[int]]:
    path = REPO_ROOT / "configs" / split / "v1.json"
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    return spec["config"], spec["seeds"]


def _episode_ids(root: Path, split: str) -> list[str]:
    d = root / split
    return sorted(p.name for p in d.iterdir() if p.is_dir() and _EP_DIR.match(p.name)) if d.is_dir() else []


def cmd_generate(split: str, episodes_root: Path = EPISODES, grader_root: Path = GRADER) -> list[str]:
    overrides, seeds = split_config(split)
    cfg = resolve(overrides)
    for root in (episodes_root, grader_root):
        for eid in _episode_ids(root, split):
            shutil.rmtree(root / split / eid)
    ids = []
    for seed in seeds:
        ep = generate_episode(seed, overrides)
        write_episode(ep, episodes_root, grader_root, split)
        ids.append(ep.episode_id)
    index = {"split": split, "version": __version__, "config_id": config_id(cfg), "episodes": sorted(ids)}
    (episodes_root / split).mkdir(parents=True, exist_ok=True)
    with open(episodes_root / split / "index.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(jsonutil.pretty(index))
    # seeds live only on the grader side
    with open(grader_root / split / "seeds.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(jsonutil.pretty({"split": split, "seeds": seeds, "config": overrides}))
    print(f"generated {len(ids)} {split} episodes ({index['config_id']})")
    return ids


def cmd_validate(split: str) -> int:
    """Independent validation of the files on disk + byte-exact regeneration check."""
    failures = 0
    overrides, seeds = split_config(split)
    eids = _episode_ids(EPISODES, split)
    if len(eids) != len(seeds):
        print(f"[{split}] expected {len(seeds)} episodes, found {len(eids)}")
        failures += 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cmd_generate_quiet(split, tmp / "e", tmp / "g")
        for eid in eids:
            open_dir, closed_dir = EPISODES / split / eid, GRADER / split / eid
            problems = []
            bundle = read_open(open_dir)
            truth, prov = read_closed(closed_dir)
            problems += [f"manifest: {e}" for e in schema.validate(bundle["manifest"], "manifest")]
            problems += [f"truth: {e}" for e in schema.validate(truth, "truth")]
            problems += invariants.compare_with_truth(bundle, truth)
            for name in OPEN_FILES:
                if (open_dir / name).read_bytes() != (tmp / "e" / split / eid / name).read_bytes():
                    problems.append(f"{name} differs from regeneration")
            for name in CLOSED_FILES:
                if (closed_dir / name).read_bytes() != (tmp / "g" / split / eid / name).read_bytes():
                    problems.append(f"{name} differs from regeneration")
            n_issues = len(truth["issues"])
            status = "OK " if not problems else "FAIL"
            print(f"[{split}] {status} {eid} issues={n_issues} ledger={bundle['manifest']['record_counts']['ledger']}"
                  f" docs={bundle['manifest']['document_count']}")
            for p in problems:
                print(f"      - {p}")
            failures += bool(problems)
    return failures


def cmd_generate_quiet(split: str, e: Path, g: Path) -> None:
    overrides, seeds = split_config(split)
    for seed in seeds:
        write_episode(generate_episode(seed, overrides), e, g, split)


def cmd_baselines(split: str, mode: str = "train", quiet: bool = False) -> list[dict]:
    eids = _episode_ids(EPISODES, split)
    summaries = []
    env = AuditEnv(mode=mode, grader_log=TRAJ / split / "grader_log.jsonl")
    (TRAJ / split).mkdir(parents=True, exist_ok=True)
    (TRAJ / split / "grader_log.jsonl").unlink(missing_ok=True)
    try:
        for pol in policies():
            rows_all = []
            for eid in eids:
                rows, summary = run_episode(env, pol, config={"episode": {"split": split, "episode_id": eid}})
                rows_all += rows
                summaries.append(summary)
            write_jsonl(rows_all, TRAJ / split / f"{pol.name}.jsonl")
    finally:
        env.close()
    with open(TRAJ / split / "summary.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(jsonutil.pretty(summaries))
    if not quiet:
        print(f"\nBaselines on {split} ({len(eids)} episodes):")
        print(format_table(summaries))
    return summaries


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
        stage(f"validate {split} (invariants, truth, byte-exact regeneration)", cmd_validate(split) == 0)
    ok = True
    for split in ("train", "dev"):
        summaries = cmd_baselines(split)
        by = {}
        for s in summaries:
            by.setdefault(s["policy"], []).append(s)
        oracle = by["oracle"]
        flag = by["flag_all"]
        ok &= all(s["reward"] >= 0.99 for s in oracle)
        ok &= sum(s["reward"] for s in flag) / len(flag) <= -0.5
        ok &= all(s["time_s"] <= 60 for s in summaries)
        ok &= all(s.get("invalid_reason") is None for s in oracle)
    stage("baselines (oracle ~ 1, flag_all low, <=60 s/episode)", ok)
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
    if args.cmd == "check":
        return cmd_check()
    return 2
