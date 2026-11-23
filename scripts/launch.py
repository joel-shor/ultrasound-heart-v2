#!/usr/bin/env python3
"""Thin CLI entrypoint to preview or submit Beaker GPU jobs (`<= 2 GPUs` enforced)."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from heartsounds.config import load_experiment_config  # noqa: E402
from heartsounds.deploy.beaker_guard import (  # noqa: E402
    LaunchPreflightError,
    build_gantry_argv,
    check_git_preflight,
    enforce_gpu_budget,
    ensure_beaker_config_for_workspace,
    executable_gantry_argv,
    load_env_with_overrides,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or submit a Beaker/Gantry GPU job (dry-run unless --submit)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hello", action="store_true", help="Run 1-GPU CUDA & /data smoke probe")
    mode.add_argument("--config", type=Path, help="Experiment YAML recipe under configs/")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--name", help="Explicit Beaker experiment name override")
    parser.add_argument("--wait", action="store_true", help="Stream logs and wait for completion")
    parser.add_argument("--submit", action="store_true", help="Submit job to Beaker")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Bypass clean-git check (dry-run)"
    )
    parser.add_argument("overrides", nargs="*", help="Optional key=value config overrides")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_path = args.config or (REPO_ROOT / "configs" / "beaker_smoke.yaml")
    cfg = load_experiment_config(cfg_path, overrides=args.overrides)
    env = load_env_with_overrides(args.env_file)

    try:
        sha = ""
        if args.submit or not args.allow_dirty:
            sha = check_git_preflight(
                REPO_ROOT,
                cfg_path if args.config else None,
                allow_dirty=args.allow_dirty,
                require_pushed=args.submit,
            )
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        run_name = args.name or f"hs2-{cfg.name}-{'hello' if args.hello else 'cv'}-{stamp}"
        rel_cfg = (
            str(cfg_path.resolve().relative_to(REPO_ROOT.resolve()))
            if args.config is not None
            else None
        )
        gantry_cmd = build_gantry_argv(
            cfg,
            env,
            mode="hello" if args.hello else "train",
            config_rel_path=rel_cfg,
            sha=sha,
            run_name=run_name,
            wait=args.wait,
            allow_dirty=args.allow_dirty and not args.submit,
            overrides=args.overrides,
        )
    except LaunchPreflightError as exc:
        print(f"[launch] Preflight failed: {exc}", file=sys.stderr)
        return 1

    print(shlex.join(gantry_cmd))
    if not args.submit:
        print("\n[launch] Dry-run only. Re-run with --submit to launch on Beaker.", file=sys.stderr)
        return 0

    try:
        active_gpus = enforce_gpu_budget(cfg.beaker.gpus, cfg.beaker.cluster)
        print(
            f"[launch] GPU budget check passed: {active_gpus} active + {cfg.beaker.gpus} requested <= 2",
            file=sys.stderr,
        )
    except LaunchPreflightError as exc:
        print(f"[launch] Blocked by GPU guard: {exc}", file=sys.stderr)
        return 1

    run_env = os.environ.copy()
    cfg_shim = ensure_beaker_config_for_workspace(env.get("BEAKER_WORKSPACE"))
    if cfg_shim is not None:
        run_env["BEAKER_CONFIG"] = str(cfg_shim)

    exec_argv = executable_gantry_argv(gantry_cmd)
    print(f"[launch] Submitting via uvx: {shlex.join(exec_argv)}", file=sys.stderr)
    return subprocess.run(exec_argv, cwd=REPO_ROOT, env=run_env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
