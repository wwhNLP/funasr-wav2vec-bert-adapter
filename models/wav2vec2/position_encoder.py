from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv1d, Module, Sequential
from typing import TYPE_CHECKING, final

from typing_extensions import override

# 【假设】你已经在一个 utils.py 文件中定义了这个
class NotSupportedError(Exception):
    """当某个功能、操作或参数不被支持时抛出。"""
    pass

# -----------------------------------------------------------------------
# 【主类】Wav2Vec 2.0 卷积位置编码器
# -----------------------------------------------------------------------
@final
class Wav2Vec2PositionEncoder(Module):
    """
    (FunASR 兼容版)
    Encodes sequences with relative positional information using a 1D convolution.
    """

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        **kwargs,  # 接收并忽略 device/dtype
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param kernel_size:
            The kernel size of the 1D convolution.
        :param num_groups:
            The number of convolution groups.
        """
        super().__init__()
        self.encoding_dim = model_dim # PositionEncoder 基类的属性

        # 【已修改】使用标准的 nn.Conv1d，移除了 fairseq2 的特殊 weight_norm
        self.conv = Conv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding=kernel_size // 2, # "same" padding
            groups=num_groups,
        )

        self.remove_pad = kernel_size % 2 == 0
        self.activation = GELU()

    @override
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        :param seqs:
            The sequences to encode. *Shape:* (B, T, C).
        :param padding_mask:
            The padding mask. *Shape:* (B, T), where `True` indicates a
            valid (non-padded) position.
        """
        if padding_mask is not None:
            # 我们必须将填充位置归零，否则噪声会泄露到卷积中
            # B, T -> B, T, 1
            seqs = seqs.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        # (B, T, C) -> (B, C, T)
        encodings = seqs.transpose(1, 2)

        # (B, C, T) -> (B, C, T)
        encodings = self.conv(encodings)

        if self.remove_pad:
            encodings = encodings[:, :, :-1]

        encodings = self.activation(encodings)

        # (B, C, T) -> (B, T, C)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings  # 残差连接

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}"


# -----------------------------------------------------------------------
# 【堆叠版】Wav2Vec 2.0 卷积位置编码器
# -----------------------------------------------------------------------
@final
class Wav2Vec2StackedPositionEncoder(Module):
    """
    (FunASR 兼容版)
    Encodes sequences with relative positional information using a stack
    of 1D convolutions.
    """

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        num_layers: int,
        **kwargs,  # 接收并忽略 device/dtype
    ) -> None:
        super().__init__()
        self.encoding_dim = model_dim # PositionEncoder 基类的属性

        k = max(3, kernel_size // num_layers)

        self.layers = Sequential()

        for _ in range(num_layers):
            layer = Wav2Vec2PositionEncoderLayer(
                model_dim,
                k,
                num_groups,
            )
            self.layers.append(layer)

    @override
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        :param seqs:
            The sequences to encode. *Shape:* (B, T, C).
        :param padding_mask:
            The padding mask. *Shape:* (B, T), where `True` indicates a
            valid (non-padded) position.
        """
        if padding_mask is not None:
            # (B, T) -> (B, T, 1)
            seqs = seqs.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        # (B, T, C) -> (B, C, T)
        encodings = seqs.transpose(1, 2)

        # (B, C, T) -> (B, C, T)
        encodings = self.layers(encodings)

        # (B, C, T) -> (B, T, C)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings  # 残差连接

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}"


# -----------------------------------------------------------------------
# 【堆叠版的辅助层】
# -----------------------------------------------------------------------
@final
class Wav2Vec2PositionEncoderLayer(Module):
    """(FunASR 兼容版)"""

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        **kwargs, # 接收并忽略 device/dtype
    ) -> None:
        super().__init__()

        self.conv = Conv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding="same", # 使用 "same" 自动处理 padding
            groups=num_groups,
        )

        # 【已修改】使用标准的 nn.LayerNorm
        self.layer_norm = nn.LayerNorm(
            model_dim, bias=True, elementwise_affine=False
        )

        self.activation = GELU()

    def forward(self, encodings: Tensor) -> Tensor:
        # (B, C, T) -> (B, C, T)
        encodings = self.conv(encodings)

        # (B, C, T) -> (B, T, C)
        encodings = encodings.transpose(1, 2)

        encodings = self.layer_norm(encodings)

        # (B, T, C) -> (B, C, T)
        encodings = encodings.transpose(1, 2)

        encodings = self.activation(encodings)

        return encodings

    if TYPE_CHECKING:
        __call__ = forward