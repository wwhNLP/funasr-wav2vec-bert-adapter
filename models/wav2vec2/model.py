#
# 【FunASR 最终兼容版】 wav2vec2/model.py
# - 完全移除 fairseq2 依赖
# - __init__ 接收来自 config.yaml 的字典参数
# - forward 签名改为 (speech, speech_lengths)
# - BatchLayout 替换为 padding_mask (Tensor)
# - 使用 PyTorch 原生组件
#

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from typing_extensions import override
# 导入 FunASR 核心组件
from funasr.register import tables

# 导入本地的 FunASR 兼容子模块
from .frontend import Wav2Vec2Frontend
from .masker import Wav2Vec2Masker
from .vector_quantizer import (
    Wav2Vec2VectorQuantizer,
    Wav2Vec2VectorQuantizerOutput,
)


@final
@tables.register("model_classes", "Wav2Vec2Model")
class Wav2Vec2Model(Module):
    """
    (FunASR 兼容版)
    Represents a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    def __init__(
        self,
        model_dim: int,
        # 子模块的配置字典
        frontend_conf: dict,
        encoder: str,           # 要加载的编码器名称 (来自 config.yaml)
        encoder_conf: dict,
        masker_conf: dict,
        quantizer_conf: dict,
        final_dim: int,
        *,
        quantizer_encoder_grad: bool = True,
        final_proj_bias: bool = True,
        num_distractors: int = 100,
        logit_temp: float = 0.1,
        **kwargs,  # 接收所有其他未声明的参数
    ) -> None:
        """
        :param frontend_conf:
            包含 Wav2Vec2Frontend 配置的字典。
        :param encoder:
            要从 FunASR 注册表中加载的编码器类的名称。
        :param encoder_conf:
            包含编码器配置的字典。
        ... (其他参数) ...
        """
        super().__init__()

        self.model_dim = model_dim

        # 在内部构建子模块
        self.encoder_frontend = Wav2Vec2Frontend(**frontend_conf)

        # 使用 FunASR 注册表动态构建编码器
        encoder_class = tables.encoder_classes.get(encoder)
        if encoder_class is None:
            raise ValueError(
                f"Encoder class '{encoder}' is not registered in funasr.tables.encoder_classes."
            )
        self.encoder = encoder_class(**encoder_conf)

        self.masker = Wav2Vec2Masker(**masker_conf)
        self.quantizer = Wav2Vec2VectorQuantizer(**quantizer_conf)

        self.quantizer_encoder_grad = quantizer_encoder_grad

        # 使用 PyTorch 原生的 nn.Linear
        self.final_proj = nn.Linear(model_dim, final_dim, final_proj_bias)

        self.final_target_proj = nn.Linear(
            self.quantizer.output_dim, final_dim, bias=True
        )

        self.num_distractors = num_distractors
        self.logit_temp = logit_temp

    def forward(
        self,
        speech: Tensor,
        speech_lengths: Tensor,
        **kwargs,
    ) -> dict:
        """
        FunASR 标准 forward 接口
        """
        features, all_hidden_states = self.extract_features(
            speech, speech_lengths
        )

        # 从 kwargs 获取权重
        diversity_weight = kwargs.get("w2v2_diversity_weight", 0.1)
        features_penalty_weight = kwargs.get("w2v2_features_penalty_weight", 10.0)
        
        output = self.quantize_and_contrast(features)

        loss = self.compute_loss(
            output,
            diversity_weight=diversity_weight,
            features_penalty_weight=features_penalty_weight,
        )
        
        # FunASR 需要一个字典
        return {
            "loss": loss.aggregate,
            "output": output,
            "all_hidden_states": all_hidden_states,
            "stats": {
                "contrastive_loss": loss.contrastive,
                "diversity_loss": loss.diversity,
                "features_penalty": loss.features_penalty,
            }
        }

    if TYPE_CHECKING:
        __call__ = forward

    def extract_features(self, speech: Tensor, speech_lengths: Tensor) -> tuple:
        """
        提取特征，同时返回所有中间层输出（用于 W2VBertModel）
        """
        features = self.run_frontend(speech, speech_lengths)
        encoder_output, all_hidden_states = self.encode_features(features)
        features.seqs = encoder_output
        return features, all_hidden_states

    def encode_features(self, features: "Wav2Vec2Features") -> tuple[Tensor, list[Tensor]]:
        ilens = features.padding_mask.sum(-1)
        encoder_out_tuple, _, _ = self.encoder(
            features.seqs,
            ilens,
            conf={"output_hidden_states": True},
        )
        return encoder_out_tuple

    def run_frontend(
        self, speech: Tensor, speech_lengths: Tensor
    ) -> Wav2Vec2Features:
        """
        (FunASR 兼容版)
        Run the encoder frontend in pretraining mode.
        """
        frontend = self.encoder_frontend

        seqs, features_lengths, raw_features = frontend.extract_features(
            speech, speech_lengths
        )
        
        # 提取特征后，长度会变化，重新计算特征的 padding_mask
        max_feat_len = seqs.size(1)
        feat_indices = torch.arange(max_feat_len, device=seqs.device).expand(seqs.size(0), -1)
        features_padding_mask = feat_indices < features_lengths.unsqueeze(1)

        # 3. 使用提取的特征作为目标
        if self.quantizer_encoder_grad:
            targets = seqs.clone()
        else:
            targets = seqs.detach().clone()

        if frontend.first_pass_dropout is not None:
            targets = frontend.first_pass_dropout(targets)

        processed_seqs, temporal_mask = frontend.process_features(
            seqs, features_padding_mask, self.masker
        )

        if temporal_mask is None:
            raise RuntimeError("`temporal_mask` is `None`.")

        targets = Wav2Vec2Masker.extract_masked_elements(targets, temporal_mask)

        return Wav2Vec2Features(processed_seqs, features_padding_mask, targets, temporal_mask, raw_features)

    def quantize_and_contrast(self, features: Wav2Vec2Features) -> Wav2Vec2Output:
        """Quantize targets and produce logits for contrastive prediction."""
        encoder_output, padding_mask, targets, temporal_mask = (
            features.seqs,
            features.padding_mask,
            features.targets,
            features.temporal_mask,
        )

        seqs = Wav2Vec2Masker.extract_masked_elements(encoder_output, temporal_mask)

        seqs = self.final_proj(seqs)

        if targets.dim() == 2:
            quantizer_input = targets.unsqueeze(0)
        else:
            quantizer_input = targets

        quantizer_output = self.quantizer(quantizer_input)

        targets = self.final_target_proj(quantizer_output.quantized_vectors)

        distractors = self._sample_distractors(targets)

        if targets.size(0) == 1 and seqs.dim() == 2:
            targets = targets.squeeze(0)
            distractors = distractors.squeeze(0)

        logits = self._compute_logits(seqs, targets, distractors)

        num_targets = logits.numel() // logits.size(-1)

        return Wav2Vec2Output(
            logits,
            targets,
            num_targets,
            temporal_mask,
            quantizer_output,
            encoder_output,
            padding_mask,
            features.raw,
        )

    def _sample_distractors(self, targets: Tensor) -> Tensor:
        batch_size, seq_len, model_dim = targets.shape
        device = targets.device

        targets_flat = targets.view(-1, model_dim)  # (N x S, M)
        indices = torch.arange(seq_len, device=device)  # (S)
        
        # 使用 torch.repeat_interleave
        indices_rep = torch.repeat_interleave(indices, repeats=self.num_distractors)  # (S x L)

        rand_indices = torch.randint(
            low=0,
            high=seq_len - 1,
            size=(batch_size, seq_len * self.num_distractors),
            device=device,
        )
        rand_indices[rand_indices >= indices_rep] += 1

        k = torch.arange(batch_size, device=device).unsqueeze(1) * seq_len
        rand_indices += k
        rand_indices = rand_indices.view(-1)

        distractors = targets_flat[rand_indices]  # (N x S x L, M)
        distractors = distractors.view(
            batch_size, seq_len, self.num_distractors, model_dim
        )

        return distractors

    def _compute_logits(
        self, seqs: Tensor, targets: Tensor, distractors: Tensor
    ) -> Tensor:
        # 修正前:
        # seqs, targets = seqs.unsqueeze(2), targets.unsqueeze(2)
        # candidates = torch.cat([targets, distractors], dim=2)
        
        # 修正后:
        # seqs: (NumMasked, Dim) -> (NumMasked, 1, Dim)
        # targets: (NumMasked, Dim) -> (NumMasked, 1, Dim)
        # distractors: (NumMasked, NumDistractors, Dim)
        # 目标是在第二个维度上拼接，所以需要将 targets 扩展一维
        seqs, targets = seqs.unsqueeze(1), targets.unsqueeze(1)
        
        # candidates 形状: (NumMasked, 1 + NumDistractors, Dim)
        candidates = torch.cat([targets, distractors], dim=1)
        
        logits = torch.cosine_similarity(seqs.float(), candidates.float(), dim=-1)

        if self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        distractor_is_target = (targets == distractors).all(-1)
        if distractor_is_target.any():
            
            logits[:, 1:][distractor_is_target] = -torch.inf

        return logits

    def compute_loss(
        self,
        output: Wav2Vec2Output,
        *,
        diversity_weight: float = 0.1,
        features_penalty_weight: float = 10.0,
    ) -> Wav2Vec2Loss:
        contrastive_loss = self.compute_contrastive_loss(output.logits)
        diversity_loss = self.compute_diversity_loss(output)
        features_penalty = self.compute_features_penalty(output)

        weighted_diversity_loss = diversity_weight * diversity_loss
        weighted_features_penalty = features_penalty_weight * features_penalty

        aggregate_loss = (
            contrastive_loss + weighted_diversity_loss + weighted_features_penalty
        )

        return Wav2Vec2Loss(
            aggregate_loss, contrastive_loss, diversity_loss, features_penalty
        )

    def compute_contrastive_loss(self, logits: Tensor) -> Tensor:
        num_masked = logits.numel() // logits.size(-1)
        logits = logits.reshape(num_masked, logits.size(-1))
        logits = logits.float()  # For numerical stability

        # 目标总是在索引 0
        targets = logits.new_zeros(num_masked, dtype=torch.int64)

        # 使用 PyTorch 原生的 F.cross_entropy
        return F.cross_entropy(logits, targets, reduction="sum")

    def compute_diversity_loss(self, output: Wav2Vec2Output) -> Tensor:
        num_entries = self.quantizer.num_codebooks * self.quantizer.num_codebook_entries
        prob_perplexity = output.quantizer_output.prob_perplexity
        quantizer_loss = (num_entries - prob_perplexity) / num_entries
        return quantizer_loss * output.num_targets

    def compute_features_penalty(self, output: Wav2Vec2Output) -> Tensor:
        raw_features = output.raw_features
        return raw_features.float().pow(2).mean() * output.num_targets

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"quantizer_encoder_grad={self.quantizer_encoder_grad}, "
            f"num_distractors={self.num_distractors}, "
            f"logit_temp={self.logit_temp:G}"
        )


# -----------------------------------------------------------------------
# Dataclass 定义
# 移除了所有 BatchLayout，替换为 Tensor 类型的 padding_mask
# -----------------------------------------------------------------------

@dataclass
class Wav2Vec2Features:
    """(FunASR 兼容版)"""
    seqs: Tensor
    padding_mask: Tensor | None
    targets: Tensor
    temporal_mask: Tensor
    raw: Tensor

@dataclass
class Wav2Vec2Output:
    """(FunASR 兼容版)"""
    logits: Tensor
    quantized_targets: Tensor
    num_targets: int
    temporal_mask: Tensor
    quantizer_output: Wav2Vec2VectorQuantizerOutput
    encoder_output: Tensor
    encoder_padding_mask: Tensor | None
    raw_features: Tensor

@dataclass
class Wav2Vec2Loss:
    aggregate: Tensor
    contrastive: Tensor
    diversity: Tensor
    features_penalty: Tensor
