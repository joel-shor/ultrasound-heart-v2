#!/usr/bin/env python3
"""CLI to curate the 4 kHz dataset, publish to GCS, and mirror to a Beaker dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from heartsounds.data.curate import curate_records, load_4khz_pickle_shards, write_curated_bundle  # noqa: E402
from heartsounds.storage.gcs import CURATED_GCS_ROOT, DEFAULT_GCP_PROJECT, upload_directory_to_gcs  # noqa: E402


def _update_dotenv_dataset(env_path: Path, dataset_id: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    replaced = False
    out_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("BEAKER_DATA_DATASET="):
            out_lines.append(f"BEAKER_DATA_DATASET={dataset_id}")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.append(f"BEAKER_DATA_DATASET={dataset_id}")
    env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curate 4 kHz HeartSounds v2 dataset.")
    parser.add_argument("--shards-dir", type=Path, default=Path("/scratch/4000sr_lenNones"))
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "v2_curated")
    parser.add_argument("--gcs-dest", default=CURATED_GCS_ROOT)
    parser.add_argument("--gcp-project", default=DEFAULT_GCP_PROJECT)
    parser.add_argument("--beaker-workspace", default="ai1/aibd-binder-pairs-db")
    parser.add_argument("--upload-gcs", action="store_true")
    parser.add_argument("--create-beaker-dataset", action="store_true")
    args = parser.parse_args(argv)

    print(f"[curate] Loading 4 kHz shards from {args.shards_dir}...", flush=True)
    records = load_4khz_pickle_shards(args.shards_dir)
    print(
        f"[curate] Loaded {len(records):,} raw rows. Deduplicating & repairing labels...",
        flush=True,
    )

    manifest_df, raw_mapping_df, waveforms_by_id = curate_records(records)
    report = write_curated_bundle(manifest_df, raw_mapping_df, waveforms_by_id, args.out_dir)
    print(json.dumps(report, indent=2), flush=True)

    if args.upload_gcs:
        print(f"[curate] Uploading curated bundle to {args.gcs_dest}...", flush=True)
        uploaded = upload_directory_to_gcs(args.out_dir, args.gcs_dest, project=args.gcp_project)
        print(f"[curate] Uploaded {len(uploaded)} files to {args.gcs_dest}", flush=True)

    if args.create_beaker_dataset:
        stage_dir = Path("/scratch/beaker_data_stage")
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)
        for fname in (
            "README.md",
            "curation_report.json",
            "manifest.parquet",
            "train_val_folds.parquet",
            "test.parquet",
            "quarantine.parquet",
        ):
            shutil.copy2(args.out_dir / fname, stage_dir / fname)

        ds_name = f"heartsounds-v2-clean-4khz-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
        print(
            f"[curate] Creating Beaker dataset {ds_name} in {args.beaker_workspace}...", flush=True
        )
        subprocess.run(
            [
                "beaker",
                "dataset",
                "create",
                str(stage_dir),
                "--name",
                ds_name,
                "--workspace",
                args.beaker_workspace,
                "--desc",
                f"Mirror of {args.gcs_dest} (4 kHz deduplicated, label-repaired fetal Doppler)",
            ],
            check=True,
        )
        res = subprocess.run(
            ["beaker", "dataset", "get", f"joel-shor/{ds_name}", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        ds_id = json.loads(res.stdout)[0]["id"]
        _update_dotenv_dataset(REPO_ROOT / ".env", ds_id)
        print(f"[curate] Created Beaker dataset {ds_id} and recorded in .env", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
