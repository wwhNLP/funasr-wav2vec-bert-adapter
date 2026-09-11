# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from fairseq2 at 7f06d6f4f5d497eec02b1a238d2071eb5dc48df3.
# See LICENSES/fairseq2.txt and THIRD_PARTY_NOTICES.md.
"""FunASR frontend with fairseq2's FBANK stacking and normalization order."""
from __future__ import annotations

import torch
from torch import nn
from .feature_extractor import Wav2Vec2FeatureExtractor, Wav2Vec2FbankFeatureExtractor
from .position_encoder import Wav2Vec2PositionEncoder


class Wav2Vec2Frontend(nn.Module):
    def __init__(self, model_dim, feature_dim, feature_extractor_conf=None,
                 pos_encoder_conf=None, layer_norm_features=False,
                 first_pass_dropout_p=0.0, final_dropout_p=0.0, dropout_p=None,
                 use_fbank=False, **kwargs):
        super().__init__()
        self.model_dim = self.output_dim = model_dim
        self.feature_dim = feature_dim
        self.use_fbank = use_fbank
        if use_fbank:
            self.feature_extractor = Wav2Vec2FbankFeatureExtractor(**(feature_extractor_conf or {}))
        elif feature_extractor_conf is not None:
            self.feature_extractor = Wav2Vec2FeatureExtractor(**feature_extractor_conf)
        else:
            self.feature_extractor = None
        if self.feature_extractor is not None and self.feature_extractor.output_dim != feature_dim:
            raise ValueError("feature_dim must match the feature extractor output dimension.")
        # Relative positional information belongs inside attention for w2v-BERT.
        if use_fbank and pos_encoder_conf is not None:
            raise ValueError("fairseq2 w2v-BERT uses relative attention, not a convolutional position frontend.")
        self.pos_encoder = (Wav2Vec2PositionEncoder(**dict(pos_encoder_conf, model_dim=model_dim))
                            if pos_encoder_conf is not None else None)
        self.post_extract_layer_norm = nn.LayerNorm(feature_dim)
        self.model_dim_proj = nn.Linear(feature_dim, model_dim) if feature_dim != model_dim else None
        self.first_pass_dropout = nn.Dropout(first_pass_dropout_p) if first_pass_dropout_p else None
        self.layer_norm = nn.LayerNorm(model_dim) if layer_norm_features else None
        dropout_p = final_dropout_p if dropout_p is None else dropout_p
        self.dropout = nn.Dropout(dropout_p) if dropout_p else None

    def extract_features(self, speech, speech_lengths):
        if self.feature_extractor is not None:
            features, lengths = self.feature_extractor(speech, speech_lengths)
        else:
            features, lengths = speech, speech_lengths
        raw_features = features.clone()
        return self.post_extract_layer_norm(features), lengths, raw_features

    def process_features(self, seqs, padding_mask, masker=None):
        if self.model_dim_proj is not None:
            seqs = self.model_dim_proj(seqs)
        if self.first_pass_dropout is not None:
            seqs = self.first_pass_dropout(seqs)
        temporal_mask = None
        if masker is not None:
            seqs, temporal_mask = masker(seqs, padding_mask)
        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask)
        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)
        if self.dropout is not None:
            seqs = self.dropout(seqs)
        return seqs, temporal_mask

    def forward(self, speech, speech_lengths, **kwargs):
        seqs, lengths, _ = self.extract_features(speech, speech_lengths)
        valid = torch.arange(seqs.size(1), device=seqs.device)[None, :] < lengths[:, None]
        seqs, _ = self.process_features(seqs, valid)
        return seqs, lengths
