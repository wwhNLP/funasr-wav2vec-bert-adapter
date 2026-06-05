#
# 【FunASR 兼容版】 frontend.py
#
# - 移除了所有 fairseq2 依赖 (BatchLayout, DataType, Linear, etc.)
# - 继承 nn.Module
# - __init__ 已重写，用于接收来自 config.yaml 的字典 (dict)
# - __init__ 已修改，在内部实例化已转换的 FeatureExtractor 和 PositionEncoder
# - forward/extract_features/process_features 签名已修改
# - BatchLayout 已被 padding_mask (Tensor) 或 features_lengths (Tensor) 替换
# - 已替换为 PyTorch 原生的 nn.Linear, nn.LayerNorm
#

from __future__ import annotations
from typing_extensions import override
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Module

# 【假设】你本地的这些文件也已经被重写为 FunASR 兼容版
from models.wav2vec2.feature_extractor import Wav2Vec2FeatureExtractor # 已转换
from models.wav2vec2.position_encoder import Wav2Vec2PositionEncoder # 已转换
from models.wav2vec2.masker import Wav2Vec2Masker # 已转换

# 【假设】你已经在一个 utils.py 文件中定义了这个
class NotSupportedError(Exception):
    """当某个功能、操作或参数不被支持时抛出。"""
    pass


@final
class Wav2Vec2Frontend(Module):
    """
    (FunASR 兼容版)
    Represents the Wav2Vec 2.0 frontend including feature extraction and
    positional encoding.
    """

    # ---------------------------------------------------------------------
    # 【重大修改 1: __init__ 签名】
    # ---------------------------------------------------------------------
    # 接收来自 config.yaml 的配置字典
    # ---------------------------------------------------------------------
    def __init__(
        self,
        model_dim: int,
        feature_dim: int,
        feature_extractor_conf: dict,
        pos_encoder_conf: dict,
        layer_norm_features: bool = True,
        first_pass_dropout_p: float = 0.0,
        final_dropout_p: float = 0.0,
        dropout_p: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.feature_dim = feature_dim
        self.output_dim = model_dim # 【修正】恢复被误删的属性

        self.feature_extractor = Wav2Vec2FeatureExtractor(**feature_extractor_conf)
        
        # 位置编码器在投影后的 model_dim 维特征上工作。
        pos_encoder_conf = dict(pos_encoder_conf)
        pos_encoder_conf["model_dim"] = model_dim
        self.pos_encoder = Wav2Vec2PositionEncoder(**pos_encoder_conf)

        if self.pos_encoder.encoding_dim != self.model_dim:
             raise ValueError(
                f"Position encoder dim ({self.pos_encoder.encoding_dim}) must match model dim ({self.model_dim})."
             )

        if self.feature_dim != model_dim:
            self.model_dim_proj = nn.Linear(self.feature_dim, model_dim)
        else:
            self.model_dim_proj = None

        self.post_extract_layer_norm = nn.LayerNorm(self.feature_dim)

        if layer_norm_features:
            self.layer_norm = nn.LayerNorm(model_dim)
        else:
            self.layer_norm = None

        if first_pass_dropout_p > 0.0:
            self.first_pass_dropout = nn.Dropout(first_pass_dropout_p)
        else:
            self.first_pass_dropout = None
            
        dropout_p = final_dropout_p if dropout_p is None else dropout_p
        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = None
            
    # ---------------------------------------------------------------------
    # 【重大修改 2: forward 签名和逻辑】
    # ---------------------------------------------------------------------
    @override
    def forward(
        self,
        speech: Tensor,          # 输入语音 (B, T_in) 或 (B, T_in, C)
        speech_lengths: Tensor,  # 语音长度 (B,)
        **kwargs,                # 接收其他参数 (可能来自 AutoModel)
    ) -> tuple[Tensor, Tensor]: # 返回 (processed_features, features_lengths)
        """
        Applies feature extraction, masking (optional), positional encoding,
        and final adjustments.

        :param speech: Input audio signal. Shape: (B, T_in) or (B, T_in, C).
        :param speech_lengths: Lengths of input audio. Shape: (B,).
        :returns:
            - Processed features ready for the encoder. Shape: (B, T_out, model_dim).
            - Lengths of the processed features. Shape: (B,).
        """
        # 1. 特征提取
        # features: (B, T_feat, feature_dim), features_lengths: (B,)
        features, features_lengths, _ = self.extract_features(speech, speech_lengths)

        # 2. 特征处理 (投影, Masking(可选), 位置编码, LayerNorm, Dropout)
        # processed_features: (B, T_feat, model_dim)
        # 注意：这里不进行 Masking，Masking 由外部调用者 (Wav2Vec2Model 或 W2VBertModel)
        # 通过调用 process_features 并传入 masker 来完成。
        processed_features, _ = self.process_features(features, None, None) # Pass None for padding_mask and masker

        return processed_features, features_lengths

    def extract_features(
        self, speech: Tensor, speech_lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        (FunASR 兼容版)
        Extracts features from the input sequences.
        :returns:
            - Normalized features. Shape: (B, T_feat, feature_dim).
            - Lengths of features. Shape: (B,).
            - Raw features (before post-LayerNorm). Shape: Same as normalized features.
        """
        if self.feature_extractor is not None:
            # feature_extractor 返回 (features, features_lengths)
            features, features_lengths = self.feature_extractor(speech, speech_lengths)
        else:
            # 如果没有内部特征提取器，假定输入已经是特征
            features = speech
            # 【重要】需要计算或传递特征长度！这里假设没有降采样。
            # 如果外部提取特征有降采样，调用者必须提供正确的 features_lengths。
            features_lengths = speech_lengths # 这是一个强假设！

        raw_features = features.clone()

        features = self.post_extract_layer_norm(features)

        return features, features_lengths, raw_features

    def process_features(
        self,
        seqs: Tensor,
        padding_mask: Tensor | None,
        masker: Wav2Vec2Masker,
    ) -> tuple[Tensor, Tensor | None]:
        if self.model_dim_proj is not None:
            seqs = self.model_dim_proj(seqs)

        if self.first_pass_dropout is not None:
            seqs = self.first_pass_dropout(seqs)

        if masker is not None:
            seqs, temporal_mask = masker(seqs, padding_mask)
        else:
            temporal_mask = None

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, temporal_mask

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        # 使用 output_dim 保持与 TransformerFrontend 基类（概念上）的一致性
        return f"feature_dim={self.feature_dim}, output_dim={self.output_dim}"
