# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
This file is a modified version of FunASR's Conformer encoder.
The key modification is the addition of the `output_hidden_states` functionality
in the ConformerEncoder's forward pass, which is essential for the w2v-BERT model
to access intermediate layer representations.
"""

from typing import Union, Dict, List, Tuple, Optional
import torch
from torch import nn

from funasr.register import tables
from funasr.models.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from funasr.models.transformer.embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
)
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.utils.nets_utils import get_activation, make_pad_mask
from funasr.models.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


class NoSubsampling(nn.Module):
    """
    A layer that just projects the input and creates a mask, without subsampling.
    This is used when `input_layer` is 'null'.
    """
    def __init__(self, idim: int, odim: int, dropout_rate: float):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
        )
        self.output_dim = odim

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.out(x)
        return x, x_mask


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""
    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=channels, bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    """Conformer Encoder layer module."""
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        stochastic_depth_rate=0.0,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)
        self.norm_mha = LayerNorm(size)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)
            self.norm_final = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.stochastic_depth_rate = stochastic_depth_rate
        self.concat_after = False # Not used in w2v-bert

    def forward(self, x_input, mask, cache=None):
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # Stochastic Depth
        skip_layer = False
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
        if skip_layer:
            return (x, pos_emb) if pos_emb is not None else x, mask

        # Macaron-style FFN
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # Multi-headed self-attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        
        if pos_emb is not None:
            x_att = self.self_attn(x, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Convolution
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # FFN
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return (x, pos_emb) if pos_emb is not None else x, mask


@tables.register("encoder_classes", "SDConformerEncoder")
class SDConformerEncoder(nn.Module):
    """
    A custom Conformer encoder that supports Stochastic Depth and returns all hidden states.
    This is necessary for the w2v-BERT model.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 31,
        stochastic_depth_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        
        # For w2v-bert, we don't need subsampling, just projection.
        if input_layer is None or input_layer == "null":
            self.embed = NoSubsampling(input_size, output_size, dropout_rate)
        else:
            raise ValueError("For w2v-bert, input_layer must be null.")

        # w2v-bert uses relative positioning handled by the attention mechanism,
        # but FunASR's conformer expects a positional encoding class.
        self.pos_enc = RelPositionalEncoding(output_size, positional_dropout_rate)
        
        self.normalize_before = normalize_before
        
        activation = get_activation("swish")
        
        # Linearly spaced stochastic depth probabilities
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_blocks)]
        
        self.encoders = nn.ModuleList([
            EncoderLayer(
                size=output_size,
                self_attn=RelPositionMultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                feed_forward=PositionwiseFeedForward(
                    output_size, linear_units, dropout_rate, activation
                ),
                feed_forward_macaron=PositionwiseFeedForward(
                    output_size, linear_units, dropout_rate, activation
                ) if macaron_style else None,
                conv_module=ConvolutionModule(
                    output_size, cnn_module_kernel, activation
                ) if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                stochastic_depth_rate=dpr[i],
            )
            for i in range(num_blocks)
        ])
        
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)
        else:
            self.after_norm = None

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        conf: dict = {},
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad, masks = self.embed(xs_pad, masks)
        
        xs_pad, pos_emb = self.pos_enc(xs_pad)

        all_hidden_states = []
        output_hidden_states = conf.get("output_hidden_states", False)

        for layer in self.encoders:
            if output_hidden_states:
                all_hidden_states.append(xs_pad)
            
            xs_pad, masks = layer((xs_pad, pos_emb), masks)
            if isinstance(xs_pad, tuple):
                xs_pad = xs_pad[0]
        
        if output_hidden_states:
            all_hidden_states.append(xs_pad)

        if self.after_norm is not None:
            xs_pad = self.after_norm(xs_pad)
        
        olens = masks.squeeze(1).sum(1)
        
        if output_hidden_states:
            return (xs_pad, all_hidden_states), olens, None
        else:
            return xs_pad, olens, None
