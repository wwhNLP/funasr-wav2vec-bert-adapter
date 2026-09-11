"""Execute pinned upstream Python definitions without importing fairseq2n.

Definitions are read from the user's FAIRSEQ2_REFERENCE checkout; dense Linear,
batch metadata and imports are supplied here. The CE boundary additionally moves
MLM logits' class axis last to fix an upstream caller/helper mismatch at this pin.
The original CE formula itself is executed unchanged. See FAIRSEQ2_ALIGNMENT.md.
torch.compile is disabled by tests.
"""
from __future__ import annotations
import ast
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import final, Protocol

import torch
from torch import nn
from torch.nn import functional as F
from typing_extensions import override

COMMIT = '7f06d6f4f5d497eec02b1a238d2071eb5dc48df3'


class BatchLayout:
    packed = False

    def __init__(self, seqs, lengths=None):
        self.width = self.max_seq_len = seqs.size(1)
        self.seq_lens = list(map(int, lengths)) if lengths is not None else [self.width] * seqs.size(0)
        self.seq_lens_pt = torch.tensor(self.seq_lens, device=seqs.device)
        self.padded = any(x != self.width for x in self.seq_lens)
        pos = torch.arange(self.width, device=seqs.device).expand(seqs.size(0), -1)
        self.position_indices = pos.masked_fill(pos >= self.seq_lens_pt[:, None], -1)

    of = classmethod(lambda cls, seqs, seq_lens=None: cls(seqs, seq_lens))


class Linear(nn.Linear):
    def __init__(self, input_dim, output_dim, bias=True, init_fn=None, device=None, dtype=None, **kwargs):
        super().__init__(input_dim, output_dim, bias=bias, device=device, dtype=dtype)
        self.input_dim, self.output_dim = input_dim, output_dim
        if init_fn is not None:
            init_fn(self)


def load_reference():
    root = Path(os.environ['FAIRSEQ2_REFERENCE'])
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f'Reference checkout must be {COMMIT}, got {actual}')
    module = types.ModuleType('_pinned_fairseq2_reference')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(torch=torch, nn=nn, Tensor=torch.Tensor, math=math, ABC=ABC,
              abstractmethod=abstractmethod, final=final, override=override,
              dataclass=dataclass, Protocol=Protocol, TYPE_CHECKING=False,
              Sequence=Sequence, OrderedDict=OrderedDict, CPU=torch.device('cpu'),
              BatchLayout=BatchLayout, Linear=Linear, ColumnShardedLinear=Linear,
              RowShardedLinear=Linear, StandardLayerNorm=nn.LayerNorm,
              TransformerFrontend=nn.Module, SequenceFeatureExtractor=nn.Module,
              TransformerEncoderLayer=nn.Module, SDPA=nn.Module,
              InternalError=RuntimeError, NotSupportedError=RuntimeError,
              InvalidOperationError=RuntimeError, RemovableHandle=torch.utils.hooks.RemovableHandle,
              pad=F.pad, softmax=F.softmax, dropout=F.dropout, gumbel_softmax=F.gumbel_softmax,
              repeat_interleave=lambda x, dim, repeat: x.repeat_interleave(repeat, dim=dim),
              unsqueeze=lambda x, dim, count: x.reshape(*x.shape, *([1] * count)),
              get_name_or_self=lambda x: x)
    for name in ('Module', 'Parameter', 'Dropout', 'ModuleList', 'GLU', 'BatchNorm1d', 'Conv1d', 'SiLU', 'ReLU'):
        ns[name] = getattr(nn, name)

    def load(path, names):
        path = root / 'src/fairseq2' / path
        parsed = ast.parse(path.read_text(), filename=str(path))
        definitions = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if {node.name for node in definitions} != set(names):
            raise ValueError(f'Missing reference definitions in {path}')
        # Preserve original function/class bodies verbatim; only remove imports.
        tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)

    load('nn/projection.py', ['init_bert_projection'])
    load('nn/utils/mask.py', ['apply_mask', 'compute_row_mask', '_compute_mask_spans', '_generate_mask'])
    load('models/wav2vec2/masker.py', ['Wav2Vec2Masker', 'StandardWav2Vec2Masker'])
    load('models/wav2vec2/feature_extractor.py', ['Wav2Vec2FbankFeatureExtractor'])
    load('models/wav2vec2/frontend.py', ['Wav2Vec2Frontend'])
    load('models/wav2vec2/vector_quantizer.py', ['Wav2Vec2VectorQuantizer', 'Wav2Vec2VectorQuantizerOutput', 'GumbelWav2Vec2VectorQuantizer', '_init_entry_projection'])
    load('models/transformer/attention_bias.py', ['AttentionBias', 'IdentityBias', 'materialize_attention_bias', '_create_block_bias_tensor', '_get_seq_ranges'])
    ns['AttentionBiasCache'] = dict
    ns['maybe_get_attention_bias_tensor'] = lambda bias, q, ql, kl, cache: ns['materialize_attention_bias'](bias, ql, kl, q.device, q.dtype)
    load('models/transformer/sdpa/naive.py', ['naive_scaled_dot_product_attention'])
    load('models/transformer/sdpa/relative.py', ['RelativePositionSDPA', 'RelativePositionalEncoding'])
    load('models/transformer/multihead_attention.py', ['MultiheadAttention', 'StandardMultiheadAttention'])
    load('models/transformer/ffn.py', ['FeedForwardNetwork', 'StandardFeedForwardNetwork'])
    load('models/conformer/convolution.py', ['ConformerConvolution'])
    load('models/conformer/block.py', ['ConformerBlock'])
    load('models/transformer/encoder.py', ['TransformerEncoder', 'TransformerEncoderLayerHook', 'StandardTransformerEncoder'])
    ns.update(log_softmax=F.log_softmax, nll_loss=F.nll_loss)
    load('nn/functional/cross_entropy.py', ['cross_entropy'])
    original_ce = ns['cross_entropy']
    ns['original_cross_entropy'] = original_ce
    load('models/wav2vec2/model.py', ['Wav2Vec2Model', 'Wav2Vec2Features', 'Wav2Vec2Output', 'Wav2Vec2Loss'])
    load('models/w2vbert/model.py', ['W2VBertModel', 'W2VBertOutput', 'W2VBertLoss'])
    def objective_ce(logits, targets, pad_idx=None, **kwargs):
        # W2VBert emits (masked positions, entries, groups), while the helper
        # expects classes last. Wav2vec2 already supplies 2-D class-last logits.
        if logits.ndim == 3:
            logits = logits.transpose(1, 2)
        return original_ce(logits, targets, pad_idx, **kwargs)
    ns['cross_entropy'] = objective_ce
    return module


def build_model(ref, cfg, bert_layers=2):
    """Wire the same upstream components as Wav2Vec2EncoderFactory's FBANK path."""
    frontend_cfg = cfg['frontend_conf']
    feature_cfg = frontend_cfg['feature_extractor_conf']
    frontend = ref.Wav2Vec2Frontend(cfg['model_dim'], frontend_cfg['feature_dim'],
        ref.Wav2Vec2FbankFeatureExtractor(feature_cfg['num_fbank_channels'], feature_cfg['stride']),
        None, layer_norm=frontend_cfg.get('layer_norm_features', False))
    ec = cfg['encoder_conf']
    dim, heads = cfg['model_dim'], ec['attention_heads']
    positions = ref.RelativePositionalEncoding(dim, ec.get('max_seq_len', 4096))
    layers = []
    for _ in range(ec['num_blocks']):
        sdpa = ref.RelativePositionSDPA(dim, heads, positions, ref.IdentityBias())
        attention = ref.StandardMultiheadAttention(dim, heads, sdpa,
            qkv_proj_init_fn=ref.init_bert_projection, output_proj_init_fn=ref.init_bert_projection)
        def ffn():
            return ref.StandardFeedForwardNetwork(dim, ec['linear_units'], True,
                inner_activation=nn.SiLU(), proj_init_fn=ref.init_bert_projection)
        layers.append(ref.ConformerBlock(nn.LayerNorm(dim), ffn(), nn.LayerNorm(dim), attention,
            nn.LayerNorm(dim), ref.ConformerConvolution(dim, ec['cnn_module_kernel']),
            nn.LayerNorm(dim), ffn(), nn.LayerNorm(dim)))
    encoder = ref.StandardTransformerEncoder(layers)
    mc = dict(cfg['masker_conf'])
    mc['temporal_span_len'] = mc.pop('temporal_mask_span_len')
    mc['spatial_span_len'] = mc.pop('spatial_mask_span_len')
    masker = ref.StandardWav2Vec2Masker(**mc)
    qc = dict(cfg['quantizer_conf'])
    qc['input_dim'], qc['output_dim'] = qc.pop('model_dim'), qc.pop('quantized_dim')
    quantizer = ref.GumbelWav2Vec2VectorQuantizer(**qc)
    wav2vec = ref.Wav2Vec2Model(dim, frontend, encoder, masker, quantizer, cfg['final_dim'],
        quantizer_encoder_grad=cfg.get('quantizer_encoder_grad', True),
        num_distractors=cfg['num_distractors'], logit_temp=cfg['logit_temp'])
    return ref.W2VBertModel(wav2vec, bert_layers, num_target_codebooks=qc['num_codebooks'])


def reference_key(key):
    key = key.replace('.encoder.encoders.', '.encoder.layers.')
    key = key.replace('.masker.mask_emb', '.masker.temporal_mask_embed')
    for name in ('r_proj', 'u_bias', 'v_bias'):
        key = key.replace(f'.self_attn.{name}', f'.self_attn.sdpa.{name}')
    return key
