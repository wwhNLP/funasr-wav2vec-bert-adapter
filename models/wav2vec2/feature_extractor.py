#
# 【FunASR 兼容版】 feature_extractor.py
#
# - Removed all fairseq2 dependencies (BatchLayout, DataType, etc.)
# - Inherits nn.Module
# - __init__ accepts standard types (will be built from config dict)
# - forward signature is now (speech, speech_lengths)
# - Replaced length calculation based on BatchLayout with calculation based on speech_lengths
# - forward returns (features, features_lengths)
# - Replaced fairseq2 modules with PyTorch equivalents (nn.LayerNorm, nn.Conv1d)
# - Removed Wav2Vec2FbankFeatureExtractor (usually not needed for raw audio pre-training)
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv1d, Dropout, GroupNorm, Module, Sequential
from torch.nn.functional import group_norm
from typing_extensions import override


# -----------------------------------------------------------------------
# Helper function to calculate output lengths after convolutions
# -----------------------------------------------------------------------
def _compute_output_lengths(
    input_lengths: Tensor, layer_descs: Sequence[tuple[int, int, int]]
) -> Tensor:
    """Calculates the output sequence lengths after applying CNN layers."""
    output_lengths = input_lengths.float()
    for _, kernel_size, stride in layer_descs:
        output_lengths = torch.floor(((output_lengths - kernel_size) / stride) + 1.0)
    return output_lengths.long()

# -----------------------------------------------------------------------
# Main Feature Extractor (Processes Raw Audio)
# -----------------------------------------------------------------------
@final
class Wav2Vec2FeatureExtractor(Module):
    """
    (FunASR Compatible)
    Extracts features from raw audio waveforms using CNNs.
    """

    def __init__(
        self,
        layer_descs: Sequence[tuple[int, int, int]],
        bias: bool,
        *,
        num_channels: int = 1,
        dropout_p: float = 0.0,
        layer_norm: bool = False, # Controls LayerNorm vs GroupNorm
        grad_scale: float = 1.0, # Note: FunASR/DeepSpeed might handle grad scaling differently
        **kwargs, # Accept extra params from config
    ) -> None:
        super().__init__()

        if not layer_descs:
            raise ValueError("`layer_descs` must not be empty.")

        self.layers = Sequential()

        if num_channels < 1:
            raise ValueError(f"`num_channels` must be >= 1, but is {num_channels}.")
        self.num_channels = num_channels
        self.output_dim = layer_descs[-1][0] # Dim after last CNN layer

        input_dim = num_channels
        for i, layer_desc in enumerate(layer_descs):
            output_dim, kernel_size, stride = layer_desc

            # 【Modified】Use PyTorch nn.LayerNorm or custom Float32GroupNorm
            if layer_norm:
                layer_norm_ = nn.LayerNorm(output_dim, elementwise_affine=True) # Standard LayerNorm
                group_norm_ = None
            elif i == 0: # GroupNorm only in the first layer if not using LayerNorm
                group_norm_ = Float32GroupNorm(output_dim, output_dim)
                layer_norm_ = None
            else:
                group_norm_ = None
                layer_norm_ = None

            layer = Wav2Vec2FeatureExtractionLayer(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                bias,
                dropout_p=dropout_p,
                group_norm=group_norm_,
                layer_norm=layer_norm_,
            )
            self.layers.append(layer)
            input_dim = output_dim

        self.layer_descs = layer_descs

        if grad_scale <= 0.0 or grad_scale > 1.0:
             # Warning: FunASR/DeepSpeed might override or conflict with this.
             # Consider removing if using external grad scaling.
            print(f"Warning: grad_scale ({grad_scale}) might conflict with FunASR/DeepSpeed's gradient handling.")
        self.grad_scale = grad_scale # Keep for now, but be aware

    @override
    def forward(
        self, speech: Tensor, speech_lengths: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Extracts features from raw audio.

        :param speech:
            Input waveforms. Shape: (B, T) or (B, T, C).
        :param speech_lengths:
            Lengths of input waveforms. Shape: (B,).
        :returns:
            - Extracted features. Shape: (B, T_out, E).
            - Lengths of features. Shape: (B,).
        """
        # Ensure input is 3D: (B, T, C) -> (B, C, T) for Conv1d
        if self.num_channels > 1:
            if speech.ndim != 3 or speech.size(2) != self.num_channels:
                 raise ValueError(f"Expected input shape (B, T, {self.num_channels}), but got {speech.shape}")
            # (B, T, C) -> (B, C, T)
            speech = speech.transpose(1, 2)
        else:
            if speech.ndim == 3 and speech.size(2) == 1:
                speech = speech.squeeze(2) # (B, T, 1) -> (B, T)
            if speech.ndim != 2:
                 raise ValueError(f"Expected input shape (B, T) or (B, T, 1), but got {speech.shape}")
            # (B, T) -> (B, 1, T)
            speech = speech.unsqueeze(1)

        # Apply CNN layers: (B, C_in, T_in) -> (B, C_out, T_out)
        features = self.layers(speech)

        # Apply gradient scaling (if enabled and training)
        if self.training and self.grad_scale != 1.0:
            # Simple scaling - might need adjustment based on how FunASR handles gradients
            features = features * self.grad_scale + features.detach() * (1.0 - self.grad_scale)

        # Transpose back: (B, C_out, T_out) -> (B, T_out, C_out)
        features = features.transpose(1, 2)

        # Calculate output lengths
        features_lengths = _compute_output_lengths(speech_lengths, self.layer_descs)

        return features, features_lengths

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"grad_scale={self.grad_scale:G}"

# -----------------------------------------------------------------------
# CNN Layer used in the Extractor
# -----------------------------------------------------------------------
@final
class Wav2Vec2FeatureExtractionLayer(Module):
    """(FunASR Compatible)"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        *,
        dropout_p: float = 0.0,
        group_norm: GroupNorm | None = None,
        layer_norm: nn.LayerNorm | None = None, # Use nn.LayerNorm
    ) -> None:
        super().__init__()

        # 【Modified】Use standard Conv1d with Kaiming init
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size,
            stride=stride,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight) # Replicate Kaiming init

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else None
        self.group_norm = group_norm
        self.layer_norm = layer_norm
        self.activation = GELU()

    def forward(self, seqs: Tensor) -> Tensor:
        # (B, C_in, T_in) -> (B, C_out, T_out)
        seqs = self.conv(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        if self.group_norm is not None:
            seqs = self.group_norm(seqs)

        if self.layer_norm is not None:
            # LayerNorm expects (B, *, C)
            seqs = seqs.transpose(1, 2) # (B, C_out, T_out) -> (B, T_out, C_out)
            seqs = self.layer_norm(seqs)
            seqs = seqs.transpose(1, 2) # (B, T_out, C_out) -> (B, C_out, T_out)

        seqs = self.activation(seqs)

        return seqs

    if TYPE_CHECKING:
        __call__ = forward

# -----------------------------------------------------------------------
# Custom GroupNorm (runs in FP32)
# -----------------------------------------------------------------------
@final
class Float32GroupNorm(nn.GroupNorm):
    """Applies Group Normalization in single-precision."""
    @override
    def forward(self, x: Tensor) -> Tensor:
        w, b = self.weight, self.bias
        fp32_x = x.float()
        fp32_w = w.float() if w is not None else None
        fp32_b = b.float() if b is not None else None
        y = group_norm(fp32_x, self.num_groups, fp32_w, fp32_b, self.eps)
        return y.type_as(x)