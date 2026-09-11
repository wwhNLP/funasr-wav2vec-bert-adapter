# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from fairseq2 at 7f06d6f4f5d497eec02b1a238d2071eb5dc48df3.
# See LICENSES/fairseq2.txt and THIRD_PARTY_NOTICES.md.
"""fairseq2 span masking, preserving an equal number of targets per utterance."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from funasr.register import tables


def compute_row_mask(shape, span_len, max_mask_prob, row_lens=None, min_num_spans=0, device=None):
    num_rows, width = shape
    if span_len <= 0 or not 0 < max_mask_prob <= 1:
        raise ValueError("Require a positive span length and mask probability in (0, 1].")
    if row_lens is None:
        row_lens = torch.full((num_rows,), width, device=device, dtype=torch.int64)
    else:
        row_lens = row_lens.to(device=device, dtype=torch.int64).view(num_rows)
    if (row_lens <= span_len).any() or (row_lens > width).any():
        raise ValueError("All feature lengths must exceed the mask span length and fit the batch width.")
    num_spans = int((max_mask_prob / span_len * (row_lens - 1)).long().min())
    if num_spans < min_num_spans or num_spans == 0:
        raise ValueError("Audio is too short for the configured minimum number of mask spans.")
    ranges = (row_lens - span_len + 1).repeat_interleave(num_spans)
    offsets = (ranges * torch.rand(num_rows * num_spans, device=device)).long().view(num_rows, -1)
    offsets = offsets.repeat_interleave(span_len, dim=-1)
    indices = torch.arange(span_len, device=device).repeat(num_spans).unsqueeze(0) + offsets
    mask = torch.zeros(shape, device=device).scatter_(1, indices, 1.0)
    count = int(torch.count_nonzero(mask, dim=-1).min())
    # Overlap can leave different counts. Randomly unmask to the batch minimum.
    scores = (torch.rand_like(mask) + 0.001) * mask
    indices = scores.topk(count, dim=1, sorted=False).indices
    return torch.zeros_like(mask, dtype=torch.bool).scatter_(1, indices, True)


@tables.register("masker_classes", "Wav2Vec2Masker")
class Wav2Vec2Masker(nn.Module):
    def __init__(self, model_dim, temporal_mask_span_len=10, max_temporal_mask_prob=0.65,
                 min_num_temporal_mask_spans=2, spatial_mask_span_len=10,
                 max_spatial_mask_prob=0.0, min_num_spatial_mask_spans=2, **kwargs):
        super().__init__()
        if not 0 < max_temporal_mask_prob <= 1:
            raise ValueError("Pretraining requires max_temporal_mask_prob in (0, 1].")
        self.model_dim = model_dim
        self.temporal_mask_span_len = temporal_mask_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob
        self.min_num_temporal_mask_spans = min_num_temporal_mask_spans
        self.spatial_mask_span_len = spatial_mask_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob
        self.min_num_spatial_mask_spans = min_num_spatial_mask_spans
        self.mask_emb = nn.Parameter(torch.empty(model_dim))
        nn.init.uniform_(self.mask_emb)

    def forward(self, seqs, padding_mask):
        batch_size, seq_len, model_dim = seqs.shape
        row_lens = None if padding_mask is None else padding_mask.sum(-1)
        temporal_mask = compute_row_mask(
            (batch_size, seq_len), self.temporal_mask_span_len, self.max_temporal_mask_prob,
            row_lens, self.min_num_temporal_mask_spans, seqs.device)
        # Like fairseq2, masking applies to both training and validation objectives.
        seqs = torch.where(temporal_mask.unsqueeze(-1), self.mask_emb.to(seqs.dtype), seqs)
        if self.max_spatial_mask_prob > 0:
            spatial = compute_row_mask(
                (batch_size, model_dim), self.spatial_mask_span_len, self.max_spatial_mask_prob,
                min_num_spans=self.min_num_spatial_mask_spans, device=seqs.device)
            # Spatial masking follows temporal masking, including at masked time steps.
            seqs = seqs.masked_fill(spatial.unsqueeze(1), 0.0)
        return seqs, temporal_mask

    @staticmethod
    def extract_masked_elements(seqs: Tensor, temporal_mask: Tensor | None) -> Tensor:
        if temporal_mask is None:
            raise ValueError("temporal_mask must be provided.")
        counts = temporal_mask.sum(-1)
        if not torch.equal(counts, counts[:1].expand_as(counts)):
            raise ValueError("Each utterance must have the same number of masked frames.")
        return seqs[temporal_mask].unflatten(0, (seqs.size(0), -1))
