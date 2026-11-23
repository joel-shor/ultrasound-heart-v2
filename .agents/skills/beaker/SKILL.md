---
name: beaker
description: Launch, monitor, and debug ultrasound-heart-v2 GPU jobs on Allen Institute Beaker clusters using beaker-gantry with a strict <= 2 concurrent GPU limit.
---

# Launching GPU Jobs on Beaker (`ultrasound-heart-v2`)

This skill documents the repository's standard workflow for running reproducible PyTorch GPU jobs on Beaker (`ai1/*` clusters) using `beaker-gantry` (`gantry run`) via `scripts/launch.py`.

The canonical skill lives at `.agents/skills/beaker/SKILL.md` (symlinked at `.claude/skills/beaker/SKILL.md`).

---

## 1. Core Guardrails & Principles

1. **Strict `<= 2 GPUs` Concurrent Limit (Hard Policy):**
   - Never request more than 2 GPUs in total across running/queued experiments in this project.
   - Single-model 5-fold CV jobs use `--gpus 1`; at most **two** 1-GPU jobs (or one 2-GPU job) may run simultaneously.
   - `scripts/launch.py` (`heartsounds.deploy.beaker_guard`) enforces `requested_gpus <= 2` and queries `beaker job list` across candidate clusters to block submission whenever `active_gpus + requested_gpus > 2`.
2. **Gantry Clones Committed GitHub History (`origin`), Not Local Disk:**
   - Every Beaker job clones the exact git SHA at `HEAD` from `origin` (`https://github.com/joel-shor/ultrasound-heart-v2`).
   - Working tree must be clean, YAML configs must be git-tracked, and `HEAD` must be pushed to `origin` before `--submit`.
3. **Separate Account/Dataset Routing (`.env`) from Experiment Recipes (`configs/*.yaml`):**
   - `.env` (gitignored; copied from `.env.example`) holds `BEAKER_WORKSPACE`, `BEAKER_BUDGET`, `BEAKER_GH_TOKEN_SECRET`, and `BEAKER_DATA_DATASET`.
   - Tracked `configs/*.yaml` files hold model architecture, training hyperparameters, cluster list, GPU count, and shared memory.
4. **Curated Dataset Mounted at `/data` + Full Provenance in GCP:**
   - Canonical curated datasets live in `gs://data-processed-dualregion/v2_curated/` (`project=heartsounds`) and are mirrored byte-for-byte into a Beaker dataset (`BEAKER_DATA_DATASET`), mounted read-only at `/data` via Gantry `--dataset`.
   - GPU containers read `/data/train_val_folds.parquet`, `/data/test.parquet`, and `/data/manifest.parquet` at local NVMe speed without requiring cloud credentials inside the worker.
5. **Dry-Run by Default & Mandatory `--hello` Probe:**
   - `python scripts/launch.py ...` prints the exact `gantry run` command and exits without submitting unless `--submit` is passed.
   - Always run a 1-GPU `--hello` probe (`python scripts/launch.py --hello --submit`) before launching full training runs when testing a new image, dataset, or cluster.

---

## 2. Credentials & `.env` Setup

1. Authenticate Beaker CLI:
   ```sh
   beaker account login
   beaker account whoami
   beaker secret list --workspace ai1/aibd-binder-pairs-db
   ```
2. Populate `.env` (copied from `.env.example`):
   ```dotenv
   BEAKER_WORKSPACE=ai1/aibd-binder-pairs-db
   BEAKER_BUDGET=ai1/aipbd-general
   BEAKER_GH_TOKEN_SECRET=GITHUB_TOKEN
   BEAKER_DATA_DATASET=<BEAKER_DATASET_ID>
   ```

---

## 3. Critical `ai1/*` Cluster & Gantry Rules

1. **Always specify a public `--docker-image` (`python:3.12-bookworm`):**
   - Gantry defaults to `ai2/conda`, which `ai1/*` clusters cannot pull. `scripts/launch.py` always emits `--docker-image python:3.12-bookworm` (or the recipe's explicit public image) and syncs `pyproject.toml` + `uv.lock` via `uv`.
2. **Never pass `--memory` on GPU clusters; only pass `--shared-memory`:**
   - Requesting system RAM via `--memory` can cause Beaker's slot allocator to reserve multiple GPU slots. `scripts/launch.py` never emits `--memory`; size `/dev/shm` via `beaker.shared_memory` (e.g., `16GiB`).
3. **Multi-Cluster Routing for Zero Queue Wait:**
   - Configure `beaker.cluster` with `[ai1/octo.ai-aws-p5en, ai1/aipbd-aws-h200, ai1/octo-hub-aws-h200, ai1/octo.ai-aws-g6e]` so Beaker schedules immediately onto whichever H200/L40S cluster has free slots.
4. **`default_org: ai1` Shim for `beaker-py` / Gantry:**
   - `beaker-py` defaults `default_org = "ai2"` if omitted in `~/.beaker/config.yml`. `beaker_guard.ensure_beaker_config_for_workspace()` automatically writes a `0600` config shim with `default_org: ai1` and invokes `uvx --from beaker-gantry==3.7.0 --with "beaker-py>=2.7.2" gantry run ...`.

---

## 4. Launching & Monitoring Commands

```sh
# 1. Preview commands (no submission)
uv run python scripts/launch.py --hello
uv run python scripts/launch.py --config configs/waveform_mil_5fold.yaml
uv run python scripts/launch.py --config configs/spec_tempogram_mil_5fold.yaml

# 2. Submit 1-GPU smoke probe (<2 min)
uv run python scripts/launch.py --hello --submit --wait

# 3. Submit neural training runs (at most 2 concurrent 1-GPU jobs)
uv run python scripts/launch.py --config configs/waveform_mil_5fold.yaml --submit
uv run python scripts/launch.py --config configs/spec_tempogram_mil_5fold.yaml --submit

# 4. Monitor & fetch results
beaker cluster list ai1
beaker workspace experiments ai1/aibd-binder-pairs-db | head -n 15
beaker experiment get <EXPERIMENT_ID>
beaker job logs <JOB_ID>
beaker dataset fetch <RESULT_DATASET_ID> --output outputs/beaker_results/<RUN_NAME>
beaker job stop <JOB_ID>
```
