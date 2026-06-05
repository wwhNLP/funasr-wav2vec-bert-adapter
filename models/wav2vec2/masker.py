#
# 【FunASR 兼容版】 wav2vec2/masker.py
#
# - 移除了所有 fairseq2 依赖 (BatchLayout, DataType, compute_row_mask 等)
# - __init__ 已重写为标准 nn.Module
# - forward 签名已修改为 (seqs: Tensor, padding_mask: Tensor)
# - 重新实现了 compute_row_mask 的核心逻辑 (现在叫 _compute_mask_from_padding_mask)
# - 保留了 W2VBertModel 依赖的 extract_masked_elements 静态方法
#

from __future__ import annotations

import math
from typing import final

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from funasr.register import tables


@final
@tables.register("masker_classes", "Wav2Vec2Masker")
class Wav2Vec2Masker(Module):
    """(FunASR 兼容版) Applies temporal masking to sequences."""

    def __init__(
        self,
        model_dim: int,
        temporal_mask_span_len: int = 10,
        max_temporal_mask_prob: float = 0.65,
        min_num_temporal_mask_spans: int = 2,
        spatial_mask_span_len: int = 10,
        max_spatial_mask_prob: float = 0.0,
        min_num_spatial_mask_spans: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.temporal_mask_span_len = temporal_mask_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob
        self.min_num_temporal_mask_spans = min_num_temporal_mask_spans
        self.spatial_mask_span_len = spatial_mask_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob
        self.min_num_spatial_mask_spans = min_num_spatial_mask_spans

        self.mask_emb = Parameter(torch.empty(model_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        torch.nn.init.uniform_(self.mask_emb)

    def forward(
        self, seqs: Tensor, padding_mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        """
        :param seqs:
            The sequences to mask. *Shape:* :math:`(N,S,M)`, where :math:`N` is the
            batch size, :math:`S` is the sequence length, and :math:`M` is the
            model dimensionality.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        """
        batch_size, seq_len, model_dim = seqs.shape

        if self.training:
            if self.max_temporal_mask_prob > 0.0:
                temporal_mask = self._compute_temporal_mask(
                    (batch_size, seq_len), padding_mask, device=seqs.device
                )
            else:
                temporal_mask = None

            if self.max_spatial_mask_prob > 0.0:
                spatial_mask = self._compute_spatial_mask(
                    (batch_size, model_dim), device=seqs.device
                )

                if spatial_mask is not None:
                    seqs = seqs.masked_fill(
                        spatial_mask.unsqueeze(1).expand_as(seqs), 0.0
                    )
            else:
                spatial_mask = None
        else:
            temporal_mask, spatial_mask = None, None

        # If `temporal_mask` is not None, it means we have to apply masking.
        if temporal_mask is not None:
            seqs[temporal_mask] = self.mask_emb

        return seqs, temporal_mask

    def _compute_temporal_mask(
        self, shape: tuple[int, int], padding_mask: Tensor | None, device: torch.device
    ) -> Tensor:
        batch_size, seq_len = shape
        
        if padding_mask is None:
            row_lens = torch.full((batch_size,), seq_len, device=device)
        else:
            row_lens = padding_mask.sum(dim=-1)

        num_spans = self._compute_num_spans(
            row_lens, self.temporal_mask_span_len, self.max_temporal_mask_prob, self.min_num_temporal_mask_spans
        )

        return self._do_compute_mask(
            shape, num_spans, self.temporal_mask_span_len, row_lens, device
        )

    def _compute_spatial_mask(
        self, shape: tuple[int, int], device: torch.device
    ) -> Tensor | None:
        batch_size, model_dim = shape

        row_lens = torch.full((batch_size,), model_dim, device=device)

        num_spans = self._compute_num_spans(
            row_lens, self.spatial_mask_span_len, self.max_spatial_mask_prob, self.min_num_spatial_mask_spans
        )

        if (num_spans == 0).all():
            return None

        return self._do_compute_mask(
            shape, num_spans, self.spatial_mask_span_len, row_lens, device
        )

    @staticmethod
    def _compute_num_spans(row_lens: Tensor, span_len: int, max_prob: float, min_num_spans: int) -> Tensor:
        num_spans = (max_prob * row_lens.float() / span_len).floor().long()
        
        # Ensure at least min_num_spans if sequence is long enough
        can_mask = (row_lens >= span_len)
        min_spans = torch.full_like(num_spans, min_num_spans)
        num_spans = torch.where(can_mask, torch.max(num_spans, min_spans), num_spans)
        
        # Don't mask if prob is 0
        if max_prob == 0.0:
            num_spans.fill_(0)

        return num_spans

    @staticmethod
    def _do_compute_mask(
        shape: tuple[int, int],
        num_spans: Tensor,
        span_len: int,
        row_lens: Tensor,
        device: torch.device,
    ) -> Tensor:
        batch_size, seq_len = shape

        mask = torch.zeros(shape, device=device, dtype=torch.bool)

        for i in range(batch_size):
            if num_spans[i] == 0:
                continue

            # A span can start at any position from 0 to seq_len - span_len.
            max_start_idx = row_lens[i].item() - span_len
            if max_start_idx < 0:
                continue

            # Sample with replacement from the valid start indices.
            span_starts = torch.randint(
                0, max_start_idx + 1, (num_spans[i].item(),), device=device
            )

            span_offsets = torch.arange(span_len, device=device).unsqueeze(0)

            # (num_spans, span_len)
            indices = span_starts.unsqueeze(1) + span_offsets

            # Un-shuffle spans to avoid overlap
            indices = indices.view(-1)
            indices, _ = torch.sort(indices)

            # Apply mask
            mask[i, indices] = True

        return mask


    @staticmethod
    def extract_masked_elements(seqs: Tensor, temporal_mask: Tensor | None) -> Tensor:
        """Extract masked elements from ``seqs``."""
        if temporal_mask is None:
            raise ValueError("`temporal_mask` must not be `None`.")

        return seqs[temporal_mask]

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"temporal_mask_span_len={self.temporal_mask_span_len}, "
            f"max_temporal_mask_prob={self.max_temporal_mask_prob}, "
            f"min_num_temporal_mask_spans={self.min_num_temporal_mask_spans}, "
            f"spatial_mask_span_len={self.spatial_mask_span_len}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}, "
            f"min_num_spatial_mask_spans={self.min_num_spatial_mask_spans}"
        )