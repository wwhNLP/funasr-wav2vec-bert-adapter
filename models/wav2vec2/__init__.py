# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# 【重大修改】导出所有必要的组件
# 这确保了当其他模块 (如 w2vbert) `from ..wav2vec2 import ...` 时，
# 能够成功找到这些类。

from .feature_extractor import Wav2Vec2FeatureExtractor
from .frontend import Wav2Vec2Frontend
from .masker import Wav2Vec2Masker
from .model import (
    Wav2Vec2Features,
    Wav2Vec2Loss,
    Wav2Vec2Model,
    Wav2Vec2Output,
)
from .position_encoder import Wav2Vec2PositionEncoder
from .vector_quantizer import (
    Wav2Vec2VectorQuantizer,
    Wav2Vec2VectorQuantizerOutput,
)

__all__ = [
    "Wav2Vec2FeatureExtractor",
    "Wav2Vec2Frontend",
    "Wav2Vec2Masker",
    "Wav2Vec2Features",
    "Wav2Vec2Loss",
    "Wav2Vec2Model",
    "Wav2Vec2Output",
    "Wav2Vec2PositionEncoder",
    "Wav2Vec2VectorQuantizer",
    "Wav2Vec2VectorQuantizerOutput",
]
