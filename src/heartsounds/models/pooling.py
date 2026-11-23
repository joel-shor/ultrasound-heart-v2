"""Masked Gated Attention MIL + Mean + Top-k temporal pooling for variable-length audio.

Designed so localized ectopic arrhythmias (e.g., 1-3 premature atrial contractions in
a 20-second clip) are captured cleanly alongside global heart rate abnormalities,
while remaining strictly invariant to batch right-padding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return boolean mask `(B, max_len)` where `True` marks valid (unpadded) frames."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1).clamp(min=1)


class MaskedGatedAttentionMILPool(nn.Module):
    """Concatenates gated-attention pool, masked mean pool, and masked top-k pool."""

    def __init__(
        self, in_dim: int, attn_dim: int = 128, topk: int = 6, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.topk = max(1, int(topk))
        self.attn_v = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
        )
        self.attn_u = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Sigmoid(),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)

    @property
    def out_dim_multiplier(self) -> int:
        return 3

    def forward(
        self,
        seq: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool `(B, C, T)` sequence using exact `lengths` `(B,)`.

        Returns:
            pooled: `(B, 3 * C)`
            attn_weights: `(B, T)` summing to 1 over valid frames
        """
        bsz, channels, t_max = seq.shape
        valid_mask = make_length_mask(lengths, t_max)  # (B, T)
        seq_bt = seq.transpose(1, 2)  # (B, T, C)

        # 1. Gated MIL attention (Ilse et al., 2018)
        gated = self.attn_v(seq_bt) * self.attn_u(seq_bt)
        scores = self.attn_w(self.attn_drop(gated)).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~valid_mask, -1.0e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), seq_bt).squeeze(1)  # (B, C)

        # 2. Masked mean pool
        mask_f = valid_mask.to(seq.dtype).unsqueeze(1)  # (B, 1, T)
        denom = mask_f.sum(dim=-1).clamp(min=1.0)
        mean_pool = (seq * mask_f).sum(dim=-1) / denom  # (B, C)

        # 3. Masked Top-k mean pool (per sample to handle clips shorter than `topk`)
        masked_seq = seq.masked_fill(~valid_mask.unsqueeze(1), -1.0e4)
        topk_list: list[torch.Tensor] = []
        for i in range(bsz):
            valid_len = int(lengths[i].clamp(min=1).item())
            k_eff = min(self.topk, valid_len, t_max)
            vals, _ = torch.topk(masked_seq[i, :, :valid_len], k=k_eff, dim=-1)
            topk_list.append(vals.mean(dim=-1))
        topk_pool = torch.stack(topk_list, dim=0)  # (B, C)

        pooled = torch.cat([attn_pool, mean_pool, topk_pool], dim=-1)
        return pooled, attn_weights
