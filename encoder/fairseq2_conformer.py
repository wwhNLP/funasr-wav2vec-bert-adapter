# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from fairseq2 at 7f06d6f4f5d497eec02b1a238d2071eb5dc48df3.
# See LICENSES/fairseq2.txt and THIRD_PARTY_NOTICES.md.
"""fairseq2 w2v-BERT Conformer, with a FunASR length/hidden-state interface."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from funasr.register import tables


def bert_linear(input_dim, output_dim):
    layer = nn.Linear(input_dim, output_dim)
    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    nn.init.zeros_(layer.bias)
    return layer


class RelativePositionAttention(nn.Module):
    def __init__(self, model_dim, num_heads, max_seq_len=4096):
        super().__init__()
        if model_dim % num_heads or model_dim % 2:
            raise ValueError("model_dim must be even and divisible by attention_heads.")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.max_seq_len = max_seq_len
        self.q_proj = bert_linear(model_dim, model_dim)
        self.k_proj = bert_linear(model_dim, model_dim)
        self.v_proj = bert_linear(model_dim, model_dim)
        self.output_proj = bert_linear(model_dim, model_dim)
        self.r_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.u_bias = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.v_bias = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_normal_(self.u_bias)
        nn.init.xavier_normal_(self.v_bias)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        inv_freq = torch.exp(torch.arange(0, model_dim, 2).float() * -math.log(10000.0) / model_dim)
        angles = torch.outer(positions, inv_freq)
        freqs = torch.empty(2 * max_seq_len - 1, model_dim)
        freqs[:max_seq_len, 0::2] = angles.flip(0).sin()
        freqs[:max_seq_len, 1::2] = angles.flip(0).cos()
        freqs[max_seq_len:, 0::2] = (-angles[1:]).sin()
        freqs[max_seq_len:, 1::2] = (-angles[1:]).cos()
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, seqs, valid):
        batch, time, dim = seqs.shape
        if time > self.max_seq_len:
            raise ValueError(f"Feature length {time} exceeds max_seq_len={self.max_seq_len}.")
        def project(layer):
            return layer(seqs).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = project(self.q_proj), project(self.k_proj), project(self.v_proj)
        r = self.freqs[self.max_seq_len - time:self.max_seq_len + time - 1].to(k.dtype)
        r = self.r_proj(r).view(1, 2 * time - 1, self.num_heads, self.head_dim).transpose(1, 2).expand(batch, -1, -1, -1)
        position = torch.matmul(q + self.v_bias[:, None], r.transpose(-1, -2))
        shifted = F.pad(position, (1, 0)).view(batch, self.num_heads, 2 * time, time)
        position = shifted[:, :, 1:].reshape_as(position)[..., :time]
        # fairseq2 scales content and relative logits separately, then adds the key mask.
        scale = self.head_dim ** -0.5
        weights = torch.matmul(q + self.u_bias[:, None], k.transpose(-1, -2)) * scale
        weights = weights + position * scale
        # Upstream padded layouts isolate both the valid block and the padding block.
        allowed = valid[:, None, :, None] == valid[:, None, None, :]
        weights = weights.masked_fill(~allowed, -torch.inf)
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to((q + self.u_bias[:, None]).dtype)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, time, dim)
        return self.output_proj(output)


class FeedForward(nn.Module):
    def __init__(self, model_dim, inner_dim, dropout_p=0.0):
        super().__init__()
        self.inner_proj = bert_linear(model_dim, inner_dim)
        self.output_proj = bert_linear(inner_dim, model_dim)
        self.inner_dropout = nn.Dropout(dropout_p)

    def forward(self, seqs):
        return self.output_proj(self.inner_dropout(F.silu(self.inner_proj(seqs))))


class ConformerConvolution(nn.Module):
    def __init__(self, model_dim, kernel_size):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(model_dim, model_dim * 2, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(model_dim, model_dim, kernel_size,
                                       padding="same", groups=model_dim, bias=False)
        self.batch_norm = nn.BatchNorm1d(model_dim)
        self.pointwise_conv2 = nn.Conv1d(model_dim, model_dim, 1, bias=False)

    def forward(self, seqs, valid):
        seqs = seqs.masked_fill(~valid.unsqueeze(-1), 0.0).transpose(1, 2)
        seqs = F.glu(self.pointwise_conv1(seqs), dim=1)
        seqs = F.silu(self.batch_norm(self.depthwise_conv(seqs)))
        return self.pointwise_conv2(seqs).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, model_dim, heads, inner_dim, kernel_size, max_seq_len, dropout_p, inner_dropout_p):
        super().__init__()
        self.ffn1_layer_norm = nn.LayerNorm(model_dim)
        self.ffn1 = FeedForward(model_dim, inner_dim, inner_dropout_p)
        self.self_attn_layer_norm = nn.LayerNorm(model_dim)
        self.self_attn = RelativePositionAttention(model_dim, heads, max_seq_len)
        self.conv_layer_norm = nn.LayerNorm(model_dim)
        self.conv = ConformerConvolution(model_dim, kernel_size)
        self.ffn2_layer_norm = nn.LayerNorm(model_dim)
        self.ffn2 = FeedForward(model_dim, inner_dim, inner_dropout_p)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, seqs, valid):
        seqs = seqs + self.dropout(self.ffn1(self.ffn1_layer_norm(seqs)) * 0.5)
        seqs = seqs + self.dropout(self.self_attn(self.self_attn_layer_norm(seqs), valid))
        seqs = seqs + self.dropout(self.conv(self.conv_layer_norm(seqs), valid))
        seqs = seqs + self.dropout(self.ffn2(self.ffn2_layer_norm(seqs)) * 0.5)
        return self.layer_norm(seqs)


@tables.register("encoder_classes", "Fairseq2ConformerEncoder")
class Fairseq2ConformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, attention_heads=16, linear_units=4096,
                 num_blocks=24, cnn_module_kernel=31, max_seq_len=4096,
                 dropout_rate=0.0, ffn_inner_dropout_rate=0.0,
                 stochastic_depth_rate=0.0, input_layer=None, **kwargs):
        super().__init__()
        if input_size != output_size or input_layer not in (None, "null"):
            raise ValueError("fairseq2 Conformer expects already projected features, without subsampling.")
        if stochastic_depth_rate:
            raise ValueError("fairseq2 w2v-BERT does not support LayerDrop/stochastic depth.")
        if kwargs.get("attention_dropout_rate", 0.0) or kwargs.get("positional_dropout_rate", 0.0):
            raise ValueError("fairseq2 relative attention has no attention/position dropout.")
        self._output_size = output_size
        self.encoders = nn.ModuleList([
            ConformerBlock(output_size, attention_heads, linear_units, cnn_module_kernel,
                           max_seq_len, dropout_rate, ffn_inner_dropout_rate)
            for _ in range(num_blocks)
        ])

    def output_size(self):
        return self._output_size

    def forward(self, xs_pad, ilens, prev_states=None, conf=None):
        valid = torch.arange(xs_pad.size(1), device=xs_pad.device)[None, :] < ilens[:, None]
        hidden = [xs_pad]
        for layer in self.encoders:
            xs_pad = layer(xs_pad, valid)
            hidden.append(xs_pad)
        output = (xs_pad, hidden) if (conf or {}).get("output_hidden_states", False) else xs_pad
        return output, ilens, None
