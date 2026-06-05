"""Small CPU/GPU smoke test for the FunASR w2v-BERT adapter."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = Path(os.environ.get("FUNASR_ROOT", REPO_ROOT.parent / "FunASR"))
if FUNASR_ROOT.exists():
    sys.path.insert(0, str(FUNASR_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from models.w2vbert.model import W2VBertModel  # noqa: E402


def main() -> None:
    cfg = dict(
        model_dim=16,
        final_dim=8,
        final_proj_bias=True,
        num_distractors=3,
        logit_temp=0.1,
        frontend_conf=dict(
            model_dim=16,
            feature_dim=8,
            feature_extractor_conf=dict(
                layer_descs=[(8, 4, 2), (8, 3, 2)],
                bias=False,
                num_channels=1,
                grad_scale=1.0,
            ),
            pos_encoder_conf=dict(model_dim=16, kernel_size=8, num_groups=2),
            layer_norm_features=True,
        ),
        encoder="SDConformerEncoder",
        encoder_conf=dict(
            input_layer=None,
            input_size=16,
            output_size=16,
            attention_heads=2,
            linear_units=32,
            num_blocks=4,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
            normalize_before=True,
            stochastic_depth_rate=0.0,
            macaron_style=True,
            use_cnn_module=True,
            cnn_module_kernel=3,
        ),
        masker_conf=dict(
            model_dim=16,
            temporal_mask_span_len=2,
            max_temporal_mask_prob=0.5,
            min_num_temporal_mask_spans=1,
            spatial_mask_span_len=2,
            max_spatial_mask_prob=0.0,
            min_num_spatial_mask_spans=1,
        ),
        quantizer_conf=dict(
            model_dim=8,
            quantized_dim=16,
            num_codebooks=1,
            num_codebook_entries=8,
            codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        ),
    )

    model = W2VBertModel(w2v2_config=cfg, num_bert_encoder_layers=2)
    model.train()

    speech = torch.randn(2, 320)
    lengths = torch.tensor([320, 300], dtype=torch.int32)
    loss, stats, weight = model(speech, lengths)
    loss.backward()

    print("loss:", float(loss.detach()))
    print("weight:", int(weight))
    print({key: float(value) for key, value in stats.items()})
    print("quantizer_entry_proj_grad:", model.w2v2_model.quantizer.entry_proj.weight.grad is not None)


if __name__ == "__main__":
    main()
