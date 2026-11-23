"""PyTorch Dataset and masked variable-length batch collation for 4 kHz waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from heartsounds.data.augment import augment_waveform, normalize_waveform
from heartsounds.schema import SUBTYPE_TO_INDEX


@dataclass(frozen=True)
class WaveformBatch:
    """Typed mini-batch produced by `masked_waveform_collate`."""

    waveforms: torch.Tensor  # float32 (B, T_max)
    lengths: torch.Tensor  # int64   (B,) valid sample counts
    labels: torch.Tensor  # float32 (B,) 0.0 or 1.0
    subtypes: torch.Tensor  # int64   (B,) index in SUBTYPE_TO_INDEX
    recording_ids: list[str]
    splits: list[str]
    patient_groups: list[str]
    curated_subtypes: list[str]


class HeartSoundsDataset(Dataset[tuple[np.ndarray, float, int, str, str, str, str]]):
    """In-memory normalized 4 kHz Doppler waveform dataset.

    During training (`is_train=True`), draws random contiguous windows of up to
    `window_seconds` and repeats minority-class or multi-crop epochs cleanly.
    During evaluation (`is_train=False`), returns full recordings (up to `max_eval_seconds`)
    so attention pooling inspects every beat in the recording.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        sample_rate: int = 4000,
        window_seconds: float = 20.0,
        max_eval_seconds: float = 60.0,
        is_train: bool = False,
        augment: bool = True,
        crops_per_recording: int = 1,
        seed: int = 42,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.window_samples = int(round(window_seconds * sample_rate))
        self.max_eval_samples = int(round(max_eval_seconds * sample_rate))
        self.is_train = bool(is_train)
        self.augment = bool(augment and is_train)
        self.crops_per_recording = max(1, int(crops_per_recording)) if is_train else 1
        self._base_seed = int(seed)

        self.recording_ids: list[str] = df["recording_id"].astype(str).tolist()
        self.splits: list[str] = df["split"].astype(str).tolist()
        self.patient_groups: list[str] = df["patient_group"].astype(str).tolist()
        self.labels: np.ndarray = df["curated_label"].to_numpy(dtype=np.float32)
        self.curated_subtypes: list[str] = df["curated_subtype"].astype(str).tolist()
        self.subtype_indices: np.ndarray = np.array(
            [SUBTYPE_TO_INDEX.get(s, 1) for s in self.curated_subtypes],
            dtype=np.int64,
        )
        self.waveforms: list[np.ndarray] = [
            normalize_waveform(np.asarray(s, dtype=np.float32)) for s in df["samples"]
        ]

    def __len__(self) -> int:
        return len(self.waveforms) * self.crops_per_recording

    def __getitem__(self, index: int) -> tuple[np.ndarray, float, int, str, str, str, str]:
        rec_idx = index % len(self.waveforms)
        wav = self.waveforms[rec_idx]

        if self.is_train:
            rng = np.random.default_rng(
                self._base_seed + index * 104729 + np.random.randint(0, 1_000_000)
            )
            if wav.size > self.window_samples:
                start = int(rng.integers(0, wav.size - self.window_samples + 1))
                wav = wav[start : start + self.window_samples]
            if self.augment:
                wav = augment_waveform(wav, rng, sample_rate=self.sample_rate)
        elif wav.size > self.max_eval_samples:
            # Center-crop ultra-long (>60s) recordings deterministically
            start = (wav.size - self.max_eval_samples) // 2
            wav = wav[start : start + self.max_eval_samples]

        return (
            wav,
            float(self.labels[rec_idx]),
            int(self.subtype_indices[rec_idx]),
            self.recording_ids[rec_idx],
            self.splits[rec_idx],
            self.patient_groups[rec_idx],
            self.curated_subtypes[rec_idx],
        )


def masked_waveform_collate(
    items: list[tuple[np.ndarray, float, int, str, str, str, str]],
) -> WaveformBatch:
    """Right-pad variable-length waveforms to batch max length and record exact lengths."""
    batch_size = len(items)
    lengths = np.array([max(1, it[0].size) for it in items], dtype=np.int64)
    max_len = int(lengths.max())

    padded = np.zeros((batch_size, max_len), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.float32)
    subtypes = np.zeros(batch_size, dtype=np.int64)
    rec_ids: list[str] = []
    splits: list[str] = []
    pgroups: list[str] = []
    c_subtypes: list[str] = []

    for i, (wav, lbl, st_idx, rid, sp, pg, cst) in enumerate(items):
        padded[i, : wav.size] = wav
        labels[i] = lbl
        subtypes[i] = st_idx
        rec_ids.append(rid)
        splits.append(sp)
        pgroups.append(pg)
        c_subtypes.append(cst)

    return WaveformBatch(
        waveforms=torch.from_numpy(padded),
        lengths=torch.from_numpy(lengths),
        labels=torch.from_numpy(labels),
        subtypes=torch.from_numpy(subtypes),
        recording_ids=rec_ids,
        splits=splits,
        patient_groups=pgroups,
        curated_subtypes=c_subtypes,
    )
