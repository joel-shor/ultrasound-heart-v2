"""Explicit registry mapping `ModelConfig.name` to PyTorch architectures."""

from __future__ import annotations

import torch.nn as nn

from heartsounds.config import ModelConfig
from heartsounds.models.spec_tempogram_mil import SpecTempogramMILNet
from heartsounds.models.waveform_mil import WaveformMILNet


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate a neural model architecture from its typed configuration."""
    if cfg.name == "waveform_mil":
        return WaveformMILNet(cfg)
    if cfg.name == "spec_tempogram_mil":
        return SpecTempogramMILNet(cfg)
    raise ValueError(
        f"Unknown model.name={cfg.name!r}; expected 'waveform_mil' or 'spec_tempogram_mil'"
    )
