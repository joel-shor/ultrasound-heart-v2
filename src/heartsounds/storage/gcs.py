"""Thin, auditable Google Cloud Storage I/O and SHA-256 checksum utilities."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

DEFAULT_GCP_PROJECT = "heartsounds"
CURATED_GCS_ROOT = "gs://data-processed-dualregion/v2_curated"


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Compute the hex SHA-256 digest of a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Parse `gs://bucket/object_path` without mangling `#` characters in filenames."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    rest = uri[len("gs://") :]
    bucket, sep, blob = rest.partition("/")
    if not bucket or not sep or not blob:
        raise ValueError(f"Expected gs://bucket/path, got {uri!r}")
    return bucket, blob


def upload_file_to_gcs(
    local_path: str | Path,
    gcs_uri: str,
    *,
    project: str = DEFAULT_GCP_PROJECT,
) -> None:
    """Upload a single file to GCS via `gcloud storage cp` (fast parallel composite uploads)."""
    src = Path(local_path)
    if not src.is_file():
        raise FileNotFoundError(f"Local file to upload does not exist: {src}")
    subprocess.run(
        ["gcloud", "storage", "cp", str(src), gcs_uri, f"--project={project}"],
        check=True,
    )


def upload_directory_to_gcs(
    local_dir: str | Path,
    gcs_prefix: str,
    *,
    project: str = DEFAULT_GCP_PROJECT,
) -> list[str]:
    """Upload all files in `local_dir` (recursively) under `gcs_prefix`."""
    root = Path(local_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {root}")
    prefix = gcs_prefix.rstrip("/")
    uploaded: list[str] = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = file_path.relative_to(root).as_posix()
        dest = f"{prefix}/{rel}"
        upload_file_to_gcs(file_path, dest, project=project)
        uploaded.append(dest)
    return uploaded
