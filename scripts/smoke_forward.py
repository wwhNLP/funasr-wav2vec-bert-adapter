"""Small CPU/GPU smoke test for the FunASR w2v-BERT adapter."""

from __future__ import annotations

import sys
import argparse
import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = Path(os.environ.get("FUNASR_ROOT", REPO_ROOT.parent / "FunASR"))
if FUNASR_ROOT.exists():
    sys.path.insert(0, str(FUNASR_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from models.w2vbert.model import W2VBertModel  # noqa: E402


def small_config() -> dict:
    return dict(
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
        encoder="Fairseq2ConformerEncoder",
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

def small_fbank_config(num_fbank_channels=80) -> dict:
    cfg = small_config()
    feature_dim = num_fbank_channels * 2
    cfg["frontend_conf"] = dict(model_dim=16, feature_dim=feature_dim, use_fbank=True,
        feature_extractor_conf=dict(num_fbank_channels=num_fbank_channels, stride=2),
        pos_encoder_conf=None, layer_norm_features=False)
    cfg["quantizer_conf"]["model_dim"] = feature_dim
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--config", help="Optional full FunASR YAML model configuration")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    if args.config:
        from omegaconf import OmegaConf
        model_conf = OmegaConf.to_container(OmegaConf.load(args.config).model_conf, resolve=True)
    else:
        model_conf = dict(w2v2_config=small_fbank_config(), num_bert_encoder_layers=2)
    model = W2VBertModel(**model_conf).to(args.device).train()
    channels = model_conf["w2v2_config"]["frontend_conf"]["feature_extractor_conf"]["num_fbank_channels"]
    speech = torch.randn(2, 300, channels, device=args.device)
    lengths = torch.tensor([300, 280], dtype=torch.int32, device=args.device)
    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.bf16):
        loss, stats, weight = model(speech, lengths)
    loss.backward()
    assert torch.isfinite(loss)
    grad = model.w2v2_model.quantizer.entry_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    print("device:", next(model.parameters()).device)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("loss:", float(loss.detach()))
    print("weight:", int(weight))
    print("quantizer_entry_proj_grad:", True)
    model.eval()
    with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.bf16):
        valid_loss, _, _ = model(speech, lengths)
    assert torch.isfinite(valid_loss)
    print("valid_loss:", float(valid_loss))
    if args.device == "cuda":
        print("peak_GiB:", torch.cuda.max_memory_allocated() / 1024**3)


if __name__ == "__main__":
    main()
