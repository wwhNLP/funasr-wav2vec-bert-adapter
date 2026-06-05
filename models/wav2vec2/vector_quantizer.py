# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

from funasr.register import tables


@final
@tables.register("quantizer_classes", "Wav2Vec2VectorQuantizer")
class Wav2Vec2VectorQuantizer(Module):
    """(FunASR 兼容版) Produces discretized representations of input vectors."""

    num_codebooks: int
    num_codebook_entries: int
    output_dim: int

    def __init__(
        self,
        model_dim: int,
        quantized_dim: int,
        num_codebooks: int,
        num_codebook_entries: int,
        codebook_sampling_temperature: tuple[float, float, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.input_dim = model_dim
        self.output_dim = quantized_dim
        self.num_codebooks = num_codebooks
        self.num_codebook_entries = num_codebook_entries

        if quantized_dim % num_codebooks != 0:
            raise ValueError(
                f"`quantized_dim` ({quantized_dim}) must be divisible by `num_codebooks` ({num_codebooks})."
            )

        num_total_entries = num_codebooks * num_codebook_entries
        self.entry_proj = nn.Linear(model_dim, num_total_entries, bias=True)

        entry_dim = quantized_dim // num_codebooks
        self.entries = Parameter(torch.empty(1, num_total_entries, entry_dim))

        self.codebook_sampling_temperature = (
            codebook_sampling_temperature
            if codebook_sampling_temperature is not None
            else (1.0, 1.0, 1.0)
        )

        self.register_buffer("num_updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("curr_temp", torch.tensor(float(self.codebook_sampling_temperature[0])))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        torch.nn.init.normal_(self.entry_proj.weight, mean=0.0, std=1.0)
        torch.nn.init.zeros_(self.entry_proj.bias)
        torch.nn.init.uniform_(self.entries)
        self.num_updates.zero_()
        self.curr_temp.fill_(float(self.codebook_sampling_temperature[0]))

    def forward(self, seqs: Tensor) -> Wav2Vec2VectorQuantizerOutput:
        """
        :param seqs:
            The sequences to quantize. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M` is
            the model dimensionality.
        """
        temp = self._compute_current_temp()

        batch_size, seq_len, _ = seqs.shape
        logits = self.entry_proj(seqs)
        logits = logits.view(batch_size * seq_len * self.num_codebooks, -1)

        _, hard_indices = logits.max(dim=-1)
        hard_probs = (
            logits.new_zeros(*logits.shape)
            .scatter_(-1, hard_indices.view(-1, 1), 1.0)
            .view(batch_size * seq_len, self.num_codebooks, -1)
        )

        avg_probs = torch.softmax(
            logits.view(batch_size * seq_len, self.num_codebooks, -1).float(),
            dim=-1,
        ).mean(dim=0)
        prob_perplexity = self._compute_prob_perplexity(avg_probs)

        if self.training:
            probs = F.gumbel_softmax(logits.float(), tau=temp, hard=True).type_as(logits)
        else:
            probs = hard_probs.view(batch_size * seq_len * self.num_codebooks, -1)

        probs = probs.view(batch_size * seq_len, -1)
        quantized_vectors = torch.sum(
            probs.view(
                batch_size * seq_len,
                self.num_codebooks,
                self.num_codebook_entries,
                1,
            )
            * self.entries.view(
                1,
                self.num_codebooks,
                self.num_codebook_entries,
                -1,
            ),
            dim=-2,
        ).view(batch_size, seq_len, -1)

        return Wav2Vec2VectorQuantizerOutput(
            quantized_vectors, probs, prob_perplexity
        )

    def _compute_current_temp(self) -> float:
        max_temp, min_temp, temp_decay = self.codebook_sampling_temperature
        temp = max_temp * (temp_decay ** int(self.num_updates))
        temp = max(temp, min_temp)
        self.curr_temp.fill_(float(temp))
        if self.training:
            self.num_updates.add_(1)
        return float(temp)

    @staticmethod
    def _compute_prob_perplexity(probs: Tensor) -> Tensor:
        return torch.exp(
            -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
        ).sum()

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_codebooks={self.num_codebooks}, "
            f"num_codebook_entries={self.num_codebook_entries}, "
            f"codebook_sampling_temperature={self.codebook_sampling_temperature}"
        )


@dataclass
class Wav2Vec2VectorQuantizerOutput:
    """Holds the output of a wav2vec 2.0 vector quantizer."""

    quantized_vectors: Tensor
    """The quantized vectors. *Shape:* :math:`(N,S,D)`, where :math:`N` is the
    batch size, :math:`S` is the sequence length, and :math:`D` is the quantized
    dimensionality."""

    cb: Tensor
    """The codebook probabilities. *Shape:* :math:`(N,S,G,V)`, where :math:`N` is
    the batch size, :math:`S` is the sequence length, :math:`G` is the number of
    codebooks, and :math:`V` is the number of entries per codebook."""

    prob_perplexity: Tensor
    """The average probability perplexity of the Gumbel-Softmax distribution."""
