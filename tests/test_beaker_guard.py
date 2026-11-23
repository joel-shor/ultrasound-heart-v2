"""Unit tests for Beaker launcher config generation and the `<= 2 GPUs` hard budget guard."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from heartsounds.config import BeakerConfig, ExperimentConfig
from heartsounds.deploy.beaker_guard import (
    LaunchPreflightError,
    build_gantry_argv,
    enforce_gpu_budget,
    ensure_beaker_config_for_workspace,
    parse_dotenv,
)


def test_parse_dotenv_and_org_shim(tmp_path: Path):
    env_p = tmp_path / ".env"
    env_p.write_text(
        "# comment\nBEAKER_WORKSPACE=ai1/aibd-binder-pairs-db\nexport BEAKER_BUDGET='ai1/aipbd-general'\n"
    )
    parsed = parse_dotenv(env_p)
    assert parsed["BEAKER_WORKSPACE"] == "ai1/aibd-binder-pairs-db"
    assert parsed["BEAKER_BUDGET"] == "ai1/aipbd-general"

    src_cfg = tmp_path / "src_beaker.yml"
    src_cfg.write_text("user_token: secret\nvolumes_address: bad\n")
    dst_cfg = tmp_path / "dst_beaker.yml"
    out = ensure_beaker_config_for_workspace(
        "ai1/aibd-binder-pairs-db", source_config=src_cfg, target_config=dst_cfg
    )
    assert out == dst_cfg
    text = dst_cfg.read_text()
    assert "default_org: ai1" in text
    assert "volumes_address:" not in text


def test_build_gantry_argv_never_emits_memory_and_mounts_data():
    cfg = ExperimentConfig(name="waveform_mil_5fold")
    env = {
        "BEAKER_WORKSPACE": "ai1/aibd-binder-pairs-db",
        "BEAKER_BUDGET": "ai1/aipbd-general",
        "BEAKER_GH_TOKEN_SECRET": "GITHUB_TOKEN",
        "BEAKER_DATA_DATASET": "01MDATASETID",
    }
    argv = build_gantry_argv(
        cfg,
        env,
        mode="train",
        config_rel_path="configs/waveform_mil_5fold.yaml",
        sha="deadbeef1234",
    )
    assert argv[:2] == ["gantry", "run"]
    assert "--memory" not in argv
    assert "--shared-memory" in argv
    assert argv[argv.index("--gpus") + 1] == "1"
    assert argv[argv.index("--dataset") + 1] == "01MDATASETID:/data"


def test_gpu_guard_blocks_more_than_2_concurrent_gpus():
    # Reject > 2 GPUs directly in config/request
    with pytest.raises(LaunchPreflightError, match="exceeds per-project maximum"):
        enforce_gpu_budget(3, ("ai1/octo.ai-aws-p5en",), author="joel-shor")

    # Simulate 2 active 1-GPU project jobs + 1 unrelated repo job (`binder-evaluation`)
    def fake_runner(cmd: list[str], **kwargs):
        cluster = cmd[cmd.index("--cluster") + 1]
        if cluster == "ai1/octo.ai-aws-p5en":
            payload = [
                {
                    "id": "job_1",
                    "status": {},
                    "limits": {"gpuCount": 1},
                    "execution": {
                        "spec": {
                            "envVars": [
                                {"name": "GITHUB_REPO", "value": "joel-shor/ultrasound-heart-v2"}
                            ]
                        }
                    },
                },
                {
                    "id": "job_other_repo",
                    "status": {},
                    "limits": {"gpuCount": 8},
                    "execution": {
                        "spec": {
                            "envVars": [
                                {
                                    "name": "GITHUB_REPO",
                                    "value": "AIBioDesign-Allen/binder-evaluation",
                                }
                            ]
                        }
                    },
                },
                {
                    "id": "job_done",
                    "status": {"finalized": "2026-11-23T00:00:00Z"},
                    "limits": {"gpuCount": 1},
                },
            ]
        else:
            payload = [
                {"id": "job_1", "status": {}, "limits": {"gpuCount": 1}},
                {"id": "job_2", "status": {}, "limits": {"gpuCount": 1}},
            ]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    # Requesting 1 more GPU when 2 are already active -> MUST raise!
    with pytest.raises(LaunchPreflightError, match="GPU budget exceeded"):
        enforce_gpu_budget(
            1,
            ("ai1/octo.ai-aws-p5en", "ai1/octo.ai-aws-g6e"),
            author="joel-shor",
            runner=fake_runner,
        )

    # When only 1 GPU is active (`ai1/octo.ai-aws-p5en` alone), requesting 1 GPU succeeds (1 + 1 <= 2)
    active = enforce_gpu_budget(
        1,
        ("ai1/octo.ai-aws-p5en",),
        author="joel-shor",
        runner=fake_runner,
    )
    assert active == 1

    # Also verify BeakerConfig > 2 fails on load
    bad_cfg = ExperimentConfig(name="bad", beaker=BeakerConfig(gpus=4))
    with pytest.raises(LaunchPreflightError):
        build_gantry_argv(
            bad_cfg,
            {"BEAKER_WORKSPACE": "ai1/ws", "BEAKER_BUDGET": "ai1/b"},
            mode="hello",
        )
