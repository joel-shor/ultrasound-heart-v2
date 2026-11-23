"""Beaker/Gantry launch preflight, `default_org` shim, and `<= 2 GPUs` enforcement."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from heartsounds.config import MAX_ALLOWED_GPUS, ExperimentConfig

RUN_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")
REQUIRED_ENV_KEYS: tuple[str, ...] = ("BEAKER_WORKSPACE", "BEAKER_BUDGET")


class LaunchPreflightError(RuntimeError):
    """Raised when environment, git state, or the `<= 2 GPUs` cap fails validation."""


def sanitize_run_name(name: str, default: str = "heartsounds-run") -> str:
    """Normalize a string into a valid Beaker experiment/task identifier."""
    cleaned = RUN_NAME_RE.sub("-", name).strip("-_")
    return cleaned or default


def parse_dotenv(path: str | Path) -> dict[str, str]:
    """Parse a simple `KEY=VALUE` `.env` file without external dependencies."""
    env_file = Path(path)
    if not env_file.is_file():
        return {}
    values: dict[str, str] = {}
    for lineno, raw in enumerate(env_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise LaunchPreflightError(f"{env_file}:{lineno}: expected KEY=VALUE")
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def load_env_with_overrides(dotenv_path: str | Path) -> dict[str, str]:
    """Load `.env` and overlay any non-empty environment variables."""
    merged = parse_dotenv(dotenv_path)
    for k, v in os.environ.items():
        if v:
            merged[k] = v
    return merged


def ensure_beaker_config_for_workspace(
    workspace: str | None,
    *,
    source_config: Path | None = None,
    target_config: Path | None = None,
) -> Path | None:
    """Create a 0600 Beaker config copy setting `default_org` to `<org>` from `<org>/<ws>`."""
    if not workspace or "/" not in workspace:
        return None
    org = workspace.split("/", 1)[0].strip()
    if not org:
        return None
    src = source_config or (Path.home() / ".beaker" / "config.yml")
    if not src.is_file():
        return None

    kept: list[str] = []
    found_org = False
    for line in src.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("volumes_address:"):
            continue
        if stripped.startswith("default_org:"):
            kept.append(f"default_org: {org}")
            found_org = True
        else:
            kept.append(line)
    if not found_org:
        kept.append(f"default_org: {org}")

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", "") or (Path.home() / ".cache"))
    dst = target_config or (cache_root / "beaker" / "heartsounds_config.yml")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    dst.chmod(0o600)
    return dst


def check_git_preflight(
    repo_root: Path,
    config_path: Path | None = None,
    *,
    allow_dirty: bool = False,
    require_pushed: bool = True,
) -> str:
    """Verify clean git tree, tracked config file, and `HEAD` pushed to a remote branch."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status and not allow_dirty:
        raise LaunchPreflightError(
            "Working tree has uncommitted changes to tracked files. "
            f"Commit and push first (or pass --allow-dirty for dry-runs):\n{status}"
        )

    if config_path is not None:
        try:
            rel_cfg = config_path.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise LaunchPreflightError(
                f"Config {config_path} must reside inside repository {repo_root}"
            ) from exc
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(rel_cfg)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0 and not allow_dirty:
            raise LaunchPreflightError(f"Config file {rel_cfg} is not tracked by git.")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    if require_pushed:
        remote_branches = subprocess.run(
            ["git", "branch", "-r", "--contains", head],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not remote_branches:
            raise LaunchPreflightError(
                f"HEAD ({head[:10]}) is not on any remote branch. Run `git push` before --submit."
            )
    return head


def resolve_beaker_author(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Query `beaker account whoami --format json` for the active Beaker username."""
    res = runner(
        ["beaker", "account", "whoami", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise LaunchPreflightError(
            f"Beaker CLI is not authenticated (`beaker account login`): {res.stderr.strip()}"
        )
    payload = json.loads(res.stdout)
    entry = payload[0] if isinstance(payload, list) else payload
    return str(entry["name"])


def _extract_job_gpus(job: Mapping[str, Any]) -> int:
    for section_name in ("limits", "requests"):
        sec = job.get(section_name)
        if isinstance(sec, Mapping) and sec.get("gpuCount") is not None:
            return int(sec["gpuCount"])
    assignment = job.get("assignmentDetails") or {}
    assigned = assignment.get("assignedResources") or {}
    if assigned.get("gpuCount") is not None:
        return int(assigned["gpuCount"])
    return 1


def count_active_user_gpus(
    clusters: Sequence[str],
    *,
    author: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Count total GPUs across all non-finalized execution jobs for `author` across `clusters`."""
    who = author or resolve_beaker_author(runner=runner)
    seen_jobs: dict[str, int] = {}
    for cluster in clusters:
        res = runner(
            [
                "beaker",
                "job",
                "list",
                "--author",
                who,
                "--cluster",
                cluster,
                "--kind",
                "execution",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise LaunchPreflightError(
                f"Failed querying active jobs on {cluster}: {res.stderr.strip()}"
            )
        jobs = json.loads(res.stdout or "[]")
        for job in jobs:
            status = job.get("status") or {}
            if status.get("finalized") or status.get("exited") or status.get("canceled"):
                continue
            job_id = str(job.get("id", ""))
            if job_id and job_id not in seen_jobs:
                seen_jobs[job_id] = _extract_job_gpus(job)
    return sum(seen_jobs.values())


def enforce_gpu_budget(
    requested_gpus: int,
    clusters: Sequence[str],
    *,
    author: str | None = None,
    max_gpus: int = MAX_ALLOWED_GPUS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Raise `LaunchPreflightError` if submitting `requested_gpus` would exceed `max_gpus`."""
    if requested_gpus < 0 or requested_gpus > max_gpus:
        raise LaunchPreflightError(
            f"Requested {requested_gpus} GPUs exceeds per-project maximum ({max_gpus})."
        )
    if requested_gpus == 0:
        return 0
    active_gpus = count_active_user_gpus(clusters, author=author, runner=runner)
    if active_gpus + requested_gpus > max_gpus:
        raise LaunchPreflightError(
            f"GPU budget exceeded: {active_gpus} active GPU(s) + {requested_gpus} requested > "
            f"{max_gpus} max concurrent GPUs allowed."
        )
    return active_gpus


def executable_gantry_argv(gantry_cmd: Sequence[str]) -> list[str]:
    """Wrap `gantry run ...` in `uvx` with `beaker-py>=2.7.2` for `ai1/*` compatibility."""
    return [
        "uvx",
        "--from",
        "beaker-gantry==3.7.0",
        "--with",
        "beaker-py>=2.7.2",
        *gantry_cmd,
    ]


def build_gantry_argv(
    cfg: ExperimentConfig,
    env: Mapping[str, str],
    *,
    mode: str = "train",
    config_rel_path: str | None = None,
    sha: str = "",
    run_name: str | None = None,
    wait: bool = False,
    allow_dirty: bool = False,
    overrides: Sequence[str] = (),
) -> list[str]:
    """Construct the exact `gantry run` command (never emitting `--memory`)."""
    missing = [k for k in REQUIRED_ENV_KEYS if not env.get(k)]
    if missing:
        raise LaunchPreflightError(
            f"Missing required environment/.env keys: {missing} (see .env.example)"
        )

    bcfg = cfg.beaker
    if bcfg.gpus > MAX_ALLOWED_GPUS:
        raise LaunchPreflightError(
            f"Config requests {bcfg.gpus} GPUs, exceeding limit of {MAX_ALLOWED_GPUS}"
        )

    exp_name = sanitize_run_name(run_name or f"hs2-{cfg.name}-{mode}")
    task_name = sanitize_run_name("hello" if mode == "hello" else bcfg.task_name)

    argv: list[str] = [
        "gantry",
        "run",
        "--name",
        exp_name,
        "--task-name",
        task_name,
        "--workspace",
        str(env["BEAKER_WORKSPACE"]),
        "--budget",
        str(env["BEAKER_BUDGET"]),
        "--priority",
        bcfg.priority,
        "--docker-image",
        bcfg.docker_image,
        "--task-timeout",
        bcfg.task_timeout,
        "--results",
        "/results",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "MPLBACKEND=Agg",
    ]
    for cluster in bcfg.cluster:
        argv += ["--cluster", cluster]
    if bcfg.gpus > 0:
        argv += ["--gpus", str(bcfg.gpus)]
    if bcfg.shared_memory:
        argv += ["--shared-memory", bcfg.shared_memory]
    if bcfg.install:
        argv += ["--install", bcfg.install]

    gh_secret = env.get("BEAKER_GH_TOKEN_SECRET") or env.get("BEAKER_GITHUB_TOKEN_SECRET")
    if gh_secret:
        argv += ["--gh-token-secret", gh_secret]

    data_dataset = env.get("BEAKER_DATA_DATASET", "").strip()
    if data_dataset:
        argv += ["--dataset", f"{data_dataset}:/data"]

    if sha:
        argv += ["--ref", sha]
    if allow_dirty:
        argv.append("--allow-dirty")

    argv.append("--yes")
    if wait:
        argv += ["--show-logs", "--timeout", "-1"]
    else:
        argv += ["--timeout", "0"]

    if mode == "hello":
        payload = ["uv", "run", "--no-sync", "python", "-m", "heartsounds.deploy.hello"]
    elif mode == "train":
        if not config_rel_path:
            raise LaunchPreflightError("mode='train' requires config_rel_path")
        payload = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "heartsounds.training.run_cv",
            "--config",
            config_rel_path,
            "--output-dir",
            "/results",
            *overrides,
        ]
    else:
        raise LaunchPreflightError(f"Unsupported launch mode: {mode!r}")

    argv.append("--")
    argv.extend(payload)
    return argv
