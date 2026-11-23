"""Unit tests for `MaskedGatedAttentionMILPool`, `WaveformMILNet`, and `SpecTempogramMILNet`."""

from __future__ import annotations

import torch

from heartsounds.config import ModelConfig
from heartsounds.models.losses import CompositeDopplerLoss
from heartsounds.models.pooling import MaskedGatedAttentionMILPool
from heartsounds.models.registry import build_model


def test_masked_gated_attention_pool_is_strictly_padding_invariant():
    torch.manual_seed(0)
    pool = MaskedGatedAttentionMILPool(in_dim=32, attn_dim=16, topk=4, dropout=0.0)
    pool.eval()

    x_unpadded = torch.randn(2, 32, 25)
    lens = torch.tensor([25, 17], dtype=torch.long)

    # Append 40 arbitrary garbage columns beyond each sample's valid length
    x_padded = torch.randn(2, 32, 65)
    x_padded[0, :, :25] = x_unpadded[0, :, :25]
    x_padded[1, :, :17] = x_unpadded[1, :, :17]

    out_short, attn_short = pool(x_unpadded, lens)
    out_long, attn_long = pool(x_padded, lens)

    assert torch.allclose(out_short, out_long, atol=1e-5)
    assert torch.allclose(attn_short[1, :17], attn_long[1, :17], atol=1e-5)
    assert float(attn_long[1, 17:].sum().item()) == 0.0


def test_waveform_and_spec_tempogram_models_forward_and_backward_cpu():
    torch.manual_seed(7)
    wavs = torch.randn(4, 16000)  # 4 seconds @ 4 kHz
    lens = torch.tensor([16000, 14000, 12000, 9600], dtype=torch.long)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    subtypes = torch.tensor([0, 2, 0, 3], dtype=torch.long)
    criterion = CompositeDopplerLoss(pos_weight=2.0)

    for model_cfg in (
        ModelConfig(name="waveform_mil", channels=32, dropout=0.1, topk_pool=3),
        ModelConfig(
            name="spec_tempogram_mil",
            channels=32,
            dropout=0.1,
            topk_pool=3,
            pretrained=False,
            backbone="resnet18",
        ),
    ):
        model = build_model(model_cfg)
        out = model(wavs, lens)
        assert out["logits"].shape == (4,)
        assert out["subtype_logits"].shape[0] == 4
        loss = criterion(out["logits"], targets, out["subtype_logits"], subtypes)
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = sum(
            float(p.grad.norm().item()) for p in model.parameters() if p.grad is not None
        )
        assert grad_norm > 0.0


def test_spec_tempogram_mask_matches_backbone_reduction_stride():
    """Ensure stride-8 backbones (e.g. resnet/seresnext) attend across the full clip, not half."""
    torch.manual_seed(11)
    wavs = torch.randn(2, 40000)  # 10s @ 4 kHz -> 501 spec frames
    lens = torch.tensor([40000, 16000], dtype=torch.long)  # 10s and 4s

    for backbone, expected_stride in (("resnet18", 8), ("convnext_tiny", 16)):
        model = build_model(
            ModelConfig(
                name="spec_tempogram_mil",
                channels=32,
                dropout=0.0,
                topk_pool=3,
                pretrained=False,
                backbone=backbone,
            )
        )
        model.eval()
        assert model.stage_stride == expected_stride
        with torch.no_grad():
            attn = model(wavs, lens)["attn_weights"]
        t_len = attn.shape[1]
        # Full-length clip must place non-zero attention weight in its second half
        assert float(attn[0, t_len // 2 :].sum().item()) > 0.1
        # Shorter clip (40% length) must place zero attention in the padded tail
        assert float(attn[1, int(t_len * 0.55) :].sum().item()) == 0.0
