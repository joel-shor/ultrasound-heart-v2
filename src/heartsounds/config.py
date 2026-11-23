"""Typed, frozen experiment configuration dataclasses and YAML loader."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

MAX_ALLOWED_GPUS = 2


@dataclass(frozen=True)
class DataConfig:
    """Dataset location, sampling rate, windowing, and DataLoader settings."""

    data_dir: str = "/data"
    sample_rate: int = 4000
    train_window_seconds: float = 20.0
    crops_per_recording_epoch: int = 2
    batch_size: int = 32
    eval_batch_size: int = 16
    num_workers: int = 4
    augment: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Architecture selection and hyperparameters."""

    name: str = "waveform_mil"
    channels: int = 128
    dropout: float = 0.25
    topk_pool: int = 6
    pretrained: bool = True
    backbone: str = "convnext_small.fb_in22k"
    init_mode: str = (
        "auto"  # auto | scratch | imagenet | audioset | doppler_ssl | audioset_doppler_ssl
    )
    audioset_repo: str = "topel/ConvNeXt-Tiny-AT"
    n_mels: int = 64
    decoupled_tempogram: bool = False
    freq_subbands: int = 1
    temporal_gru: bool = False
    inst_freq_channel: bool = False


@dataclass(frozen=True)
class OptimConfig:
    """Optimization, loss weighting, regularization, and schedule parameters."""

    lr: float = 1.0e-3
    backbone_lr_scale: float = 0.25
    weight_decay: float = 1.0e-2
    epochs: int = 28
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    focal_gamma: float = 1.5
    pos_weight_scale: float = 0.35
    auc_loss_weight: float = 0.35
    subtype_loss_weight: float = 0.15
    mixup_alpha: float = 0.2
    early_stopping_patience: int = 8
    mixed_precision: bool = True
    ssl_epochs: int = 0
    ssl_lr: float = 1.0e-3
    ssl_mask_ratio: float = 0.60


@dataclass(frozen=True)
class BeakerConfig:
    """Beaker GPU execution specification (account/secrets stay in gitignored .env)."""

    cluster: tuple[str, ...] = (
        "ai1/octo.ai-aws-p5en",
        "ai1/aipbd-aws-h200",
        "ai1/octo-hub-aws-h200",
        "ai1/octo.ai-aws-g6e",
    )
    gpus: int = 1
    priority: str = "high"
    shared_memory: str = "16GiB"
    docker_image: str = "python:3.12-bookworm"
    task_timeout: str = "6h"
    task_name: str = "train"
    install: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, immutable specification of a training/evaluation run."""

    name: str = "base"
    seed: int = 42
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    output_dir: str = "outputs/runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    beaker: BeakerConfig = field(default_factory=BeakerConfig)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["folds"] = list(self.folds)
        raw["beaker"]["cluster"] = list(self.beaker.cluster)
        return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _apply_cli_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected key=value override, got: {item!r}")
        dotted_key, _, raw_val = item.partition("=")
        parsed_val = yaml.safe_load(raw_val)
        parts = dotted_key.strip().split(".")
        cursor = updated
        for p in parts[:-1]:
            if p not in cursor or not isinstance(cursor[p], dict):
                cursor[p] = {}
            cursor = cursor[p]
        cursor[parts[-1]] = parsed_val
    return updated


def _build_section(cls: type, values: dict[str, Any]) -> Any:
    valid_names = {f.name for f in fields(cls)}
    unknown = set(values.keys()) - valid_names
    if unknown:
        raise ValueError(f"Unknown keys {sorted(unknown)} for {cls.__name__}")
    coerced = dict(values)
    if cls is BeakerConfig and "cluster" in coerced:
        clusters = coerced["cluster"]
        if isinstance(clusters, str):
            clusters = [clusters]
        coerced["cluster"] = tuple(str(c) for c in clusters)
    return cls(**coerced)


def load_experiment_config(
    config_path: str | Path,
    *,
    base_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load an `ExperimentConfig`, inheriting from `configs/base.yaml` when present."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    if base_path is None:
        candidate = cfg_file.parent / "base.yaml"
        if candidate.is_file() and candidate.resolve() != cfg_file.resolve():
            base_path = candidate

    raw: dict[str, Any] = {}
    if base_path is not None and Path(base_path).is_file():
        loaded_base = yaml.safe_load(Path(base_path).read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, loaded_base)

    loaded_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded_cfg, dict):
        raise ValueError(f"Expected YAML mapping in {cfg_file}")
    raw = _deep_merge(raw, loaded_cfg)

    if overrides:
        raw = _apply_cli_overrides(raw, overrides)

    top_valid = {f.name for f in fields(ExperimentConfig)}
    unknown_top = set(raw.keys()) - top_valid
    if unknown_top:
        raise ValueError(f"Unknown top-level config keys {sorted(unknown_top)} in {cfg_file}")

    folds_raw = raw.get("folds", [0, 1, 2, 3, 4])
    beaker_cfg = _build_section(BeakerConfig, raw.get("beaker") or {})
    if beaker_cfg.gpus < 0 or beaker_cfg.gpus > MAX_ALLOWED_GPUS:
        raise ValueError(
            f"beaker.gpus={beaker_cfg.gpus} violates repository policy (must be in 0..{MAX_ALLOWED_GPUS})"
        )

    return ExperimentConfig(
        name=str(raw.get("name", cfg_file.stem)),
        seed=int(raw.get("seed", 42)),
        folds=tuple(int(f) for f in folds_raw),
        output_dir=str(raw.get("output_dir", "outputs/runs")),
        data=_build_section(DataConfig, raw.get("data") or {}),
        model=_build_section(ModelConfig, raw.get("model") or {}),
        optim=_build_section(OptimConfig, raw.get("optim") or {}),
        beaker=beaker_cfg,
    )
