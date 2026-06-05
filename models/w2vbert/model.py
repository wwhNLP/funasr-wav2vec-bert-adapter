# -------------------------------------------------------------------------
# 1. 导入所有必需的库
# -------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from funasr.register import tables

# 导入你本地重构好的、FunASR兼容的wav2vec2组件
from models.wav2vec2 import (
    Wav2Vec2Loss,
    Wav2Vec2Masker,
    Wav2Vec2Model,
    Wav2Vec2Output,
    Wav2Vec2VectorQuantizerOutput,
)


# -------------------------------------------------------------------------
# 2. 模型定义
# -------------------------------------------------------------------------
@final
@tables.register("model_classes", "W2VBertModel")
class W2VBertModel(Module):
    """
    (FunASR 最终兼容版)
    将一个已注册的 Wav2Vec2Model 包装为 w2v-BERT 模型。
    """

    def __init__(
        self,
        w2v2_config: dict,
        num_bert_encoder_layers: int,
        num_target_codebooks: int = 1,
        **kwargs,
    ) -> None:
        """
        :param w2v2_config:
            一个包含 Wav2Vec2Model 所有配置的字典 (来自 config.yaml)。
        :param num_bert_encoder_layers:
            用于 BERT 任务的 Transformer 编码器层数。
        :param num_target_codebooks:
            用于 BERT 任务的目标码本数量。
        """
        super().__init__()

        # 1. 在内部构建 FunASR 兼容的 Wav2Vec2Model
        self.w2v2_model = Wav2Vec2Model(**w2v2_config)

        # 2. 保存核心参数
        self.model_dim = self.w2v2_model.model_dim
        self.num_bert_encoder_layers = num_bert_encoder_layers
        self.num_target_codebooks = num_target_codebooks

        # 3. 使用 PyTorch 原生的 Linear 作为 BERT 投影层
        self.final_bert_proj = nn.Linear(
            self.model_dim,
            self.w2v2_model.quantizer.num_codebook_entries * num_target_codebooks,
            bias=True,
        )

    @override
    def forward(
        self,
        speech: Tensor,
        speech_lengths: Tensor,
        **kwargs,
    ) -> tuple[Tensor, dict, Tensor]:
        """
        FunASR 标准前向传播函数。
        """
        bert_weight = kwargs.get("bert_weight", 1.0)
        bert_label_smoothing = kwargs.get("bert_label_smoothing", 0.0)
        w2v2_weight = kwargs.get("w2v2_weight", 1.0)
        w2v2_diversity_weight = kwargs.get("w2v2_diversity_weight", 0.1)
        w2v2_features_penalty_weight = kwargs.get("w2v2_features_penalty_weight", 10.0)

        w2v2_features = self.w2v2_model.run_frontend(speech, speech_lengths)
        encoder_output, all_hidden_states = self.w2v2_model.encode_features(w2v2_features)

        num_total_layers = len(all_hidden_states)
        bert_input_layer_idx = num_total_layers - self.num_bert_encoder_layers - 1
        
        if bert_input_layer_idx < 0:
            raise ValueError(f"num_bert_encoder_layers is too large.")

        w2v2_features.seqs = all_hidden_states[bert_input_layer_idx]
        w2v2_output = self.w2v2_model.quantize_and_contrast(w2v2_features)
        temporal_mask = w2v2_output.temporal_mask

        masked_bert_input = Wav2Vec2Masker.extract_masked_elements(
            encoder_output, temporal_mask
        )
        bert_logits = self.final_bert_proj(masked_bert_input)
        bert_logits = bert_logits.view(
            -1,
            self.w2v2_model.quantizer.num_codebook_entries,
            self.num_target_codebooks,
        )
        
        bert_targets = self._get_target_indices(w2v2_output.quantizer_output)
        
        # 5. 组合输出和计算总损失
        output = W2VBertOutput(w2v2_output, bert_logits, bert_targets)
        loss = self.compute_loss(
            output,
            bert_weight=bert_weight,
            bert_label_smoothing=bert_label_smoothing,
            w2v2_weight=w2v2_weight,
            w2v2_diversity_weight=w2v2_diversity_weight,
            w2v2_features_penalty_weight=w2v2_features_penalty_weight,
        )

        # 6. 返回 FunASR 期望的元组
        stats = dict(
            bert_loss=loss.bert.detach(),
            w2v2_contrastive_loss=loss.w2v2.contrastive.detach(),
            w2v2_diversity_loss=loss.w2v2.diversity.detach(),
            w2v2_features_penalty=loss.w2v2.features_penalty.detach(),
            w2v2_aggregate_loss=loss.w2v2.aggregate.detach(),
        )
        weight = torch.tensor(speech.size(0), device=loss.aggregate.device)

        # NaN loss detection
        if torch.isnan(loss.aggregate):
            import torch.distributed as dist
            print(f"!!!!!!!!!!!!! NaN loss detected on rank {dist.get_rank()} !!!!!!!!!!!!!")
            print(f"bert_loss: {loss.bert.item()}")
            print(f"w2v2_contrastive_loss: {loss.w2v2.contrastive.item()}")
            print(f"w2v2_diversity_loss: {loss.w2v2.diversity.item()}")
            raise RuntimeError(f"NaN loss detected on rank {dist.get_rank()}. Stopping training.")

        return loss.aggregate, stats, weight

    def _get_target_indices(
        self, quantizer_output: Wav2Vec2VectorQuantizerOutput
    ) -> Tensor:
        num_codebooks = self.w2v2_model.quantizer.num_codebooks
        batch_size, seq_len = quantizer_output.quantized_vectors.shape[:2]
        cb = quantizer_output.cb.view(batch_size * seq_len * num_codebooks, -1)
        indices = cb.argmax(dim=-1).view(-1, num_codebooks)
        indices = indices[..., : self.num_target_codebooks]
        return indices.detach()

    def compute_loss(
        self,
        output: W2VBertOutput,
        *,
        bert_weight: float = 1.0,
        bert_label_smoothing: float = 0.0,
        w2v2_weight: float = 1.0,
        w2v2_diversity_weight: float = 0.1,
        w2v2_features_penalty_weight: float = 10.0,
    ) -> W2VBertLoss:
        # --- BERT 损失计算 (使用 PyTorch 原生 CE) ---
        # 调整 logits 和 targets 的形状以匹配 F.cross_entropy
        bert_logits_permuted = output.bert_logits.permute(0, 2, 1)
        bert_logits_reshaped = bert_logits_permuted.reshape(
            -1, bert_logits_permuted.size(-1)
        )
        bert_targets_reshaped = output.bert_targets.reshape(-1)
        
        bert_loss = F.cross_entropy(
            bert_logits_reshaped,
            bert_targets_reshaped,
            reduction="sum",
            label_smoothing=bert_label_smoothing,
        )
        
        # --- wav2vec 2.0 损失计算 (调用子模块) ---
        w2v2_loss = self.w2v2_model.compute_loss(
            output.w2v2_output,
            diversity_weight=w2v2_diversity_weight,
            features_penalty_weight=w2v2_features_penalty_weight,
        )

        # --- 组合损失 ---
        weighted_bert_loss = bert_weight * bert_loss
        weighted_w2v2_loss = w2v2_weight * w2v2_loss.aggregate

        return W2VBertLoss(
            weighted_bert_loss + weighted_w2v2_loss, bert_loss, w2v2_loss
        )

    @override
    def extra_repr(self) -> str:
        return (
            f"model_dim={self.model_dim}, "
            f"num_bert_encoder_layers={self.num_bert_encoder_layers}, "
            f"num_target_codebooks={self.num_target_codebooks}"
        )


# -------------------------------------------------------------------------
# 3. 数据类 (不依赖 fairseq2，可以保留)
# -------------------------------------------------------------------------
@dataclass
class W2VBertOutput:
    w2v2_output: Wav2Vec2Output
    bert_logits: Tensor
    bert_targets: Tensor

@dataclass
class W2VBertLoss:
    aggregate: Tensor
    bert: Tensor
    w2v2: Wav2Vec2Loss
