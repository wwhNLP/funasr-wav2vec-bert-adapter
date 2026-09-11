# Third-Party Notices

This project adapts code and ideas from:

- FunASR, licensed under the MIT License.
- Meta fairseq2 wav2vec2 / w2v-BERT, pinned at
  `7f06d6f4f5d497eec02b1a238d2071eb5dc48df3`. The upstream root license at
  this commit is reproduced in [LICENSES/fairseq2.txt](LICENSES/fairseq2.txt).
  Some upstream source headers retain the older BSD-style wording.

The fairseq2 adaptations include FBANK stacking, frontend normalization, masking,
relative attention, Conformer blocks, quantization and SSL loss formulas.
The optional reference tests execute definitions from a separate upstream checkout.
See [FAIRSEQ2_ALIGNMENT.md](FAIRSEQ2_ALIGNMENT.md) for their interface shim.
Keep original copyright notices when redistributing or modifying derived code.
