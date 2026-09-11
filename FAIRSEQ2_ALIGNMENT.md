# fairseq2 w2v-BERT alignment

Reference: [facebookresearch/fairseq2 at 7f06d6f4](https://github.com/facebookresearch/fairseq2/tree/7f06d6f4f5d497eec02b1a238d2071eb5dc48df3),
full commit `7f06d6f4f5d497eec02b1a238d2071eb5dc48df3` (0.9.0.dev0).
This is the implemented fairseq2 variant, not a claim to reproduce the original
paper's complete training setup.

## Changes from the previous adapter

| Component | Current default behavior |
| --- | --- |
| Features | 80-channel Kaldi native FBANK, stack 2 frames into 160 channels; no waveform CNN or convolutional position frontend |
| Frontend | Feature LayerNorm, projection 160→1024, no extra post-projection LayerNorm by default |
| Encoder | 24 Conformer blocks, width 1024, 16 heads, FFN width 4096, convolution kernel 31; no input scaling/projection or outer final norm |
| Block | Half-step SiLU FFN → relative self-attention → GLU/depthwise convolution/BatchNorm/SiLU → half-step FFN → final LayerNorm; branch inputs are pre-normalized |
| Padding | Attention isolates valid and padded blocks; convolution zeros padded inputs before processing, following the pinned implementation |
| Mask | Span length 10, probability .65, at least 2 spans; upstream random span sampling and equal masked-frame counts within a batch |
| Negatives | 100 masked targets sampled within the same utterance, excluding the positive position; identical vectors receive −∞ logits |
| Quantizer | Input 160, output 1024, one group with 1024 entries; hard straight-through Gumbel sampling |
| Objectives | Layer 8 contrastive objective and layer 24 MLM objective; final contrastive dimension 768, temperature .1 |
| Reduction | Sum of contrastive + .1 × diversity + 10 × feature penalty + MLM; optimizer gradients divided by total masked positions across accumulated microbatches/ranks |

Gumbel temperature remains `max(.1, 2 × .999995**num_updates)`, with
`num_updates` incremented on each training forward, as in this fairseq2 version.
MLM integer targets are detached. Label smoothing uses fairseq's mass over the
incorrect classes (`epsilon / (V - 1)`), which differs from PyTorch's built-in CE.
Dropout and LayerDrop are disabled in both supplied model configurations.

The waveform converter uses the Python binding of Kaldi native FBANK, the same
backend used by fairseq2's native converter. Default scale is 1 and standardization
is disabled. Native fairseq2n preprocessing was not executed for a bitwise comparison.
Batch budgets and waveform length limits still use raw audio samples; only the
collated model inputs and their lengths use FBANK frames.

## Upstream interface corrections

At this pin, `models/w2vbert/model.py` emits MLM logits as `(positions, entries,
groups)`, but `nn/functional/cross_entropy.py` expects classes last. Passing these
tensors directly to that helper fails for the default one-group configuration.
The adapter transposes to `(positions, groups, entries)` before evaluating the
original mathematical objective. The reference test harness applies this same
single boundary correction, then executes the original upstream CE implementation.
It does not claim that the unmodified upstream entry point runs successfully.

The upstream `_get_target_indices` also slices by the total codebook count instead
of `num_target_codebooks`. The adapter retains the requested target-group slice.
Numerical end-to-end reference tests use equal total/target group counts (including
2 groups); the default configuration uses 1 for both.

## Reproduce validation

Install `requirements.txt`, then prepare the reference checkout:

```bash
git clone https://github.com/facebookresearch/fairseq2.git /tmp/fairseq2-w2vbert-reference
git -C /tmp/fairseq2-w2vbert-reference checkout 7f06d6f4f5d497eec02b1a238d2071eb5dc48df3
FAIRSEQ2_REFERENCE=/tmp/fairseq2-w2vbert-reference CUDA_VISIBLE_DEVICES=4 \
  python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=4 python scripts/smoke_forward.py \
  --device cuda --bf16 --config configs/w2vbert_pretrain.yaml
```

`tests/fairseq2_reference.py` loads the original Python definitions from the pinned
checkout using AST, supplies dense Linear and batch-layout/import plumbing, and
disables `torch.compile`. It executes the original frontend, mask generator,
relative attention, Conformer, quantizer and objective definitions with the CE
boundary correction above. It does not import the full fairseq2 runtime or exercise
its distributed trainer. Without `FAIRSEQ2_REFERENCE`, upstream comparisons skip.

Validation on GPU 4 with PyTorch/torchaudio 2.11.0+cu128, FunASR 1.4.15 and
kaldi-native-fbank 1.22.3:

- Full regression suite: 16 tests passed, including actual launcher training,
  checkpoint resume, data loading/resampling and mixed precision.
- Final 8 algorithm/reduction/reference tests passed again after changing MLM CE
  to the original fairseq smoothing formula.
- Identical seeded masks; copied parameters produce matching CPU/GPU FP32 train
  and eval losses, intermediate encoder output, contrastive logits, input gradients
  and every parameter gradient. Loss/output tolerances: rtol/atol 2e-5;
  gradient tolerances: rtol 5e-4, atol 5e-5. TF32 disabled for these comparisons.
- BF16 reference loss and quantizer projection gradient comparisons pass at
  rtol/atol 1e-3 and 1e-2 respectively.
- Unequal microbatch target counts produce the same optimizer update as the
  combined target-normalized objective; validation uses target-weighted loss.
- Full default model: 609,433,408 parameters; BF16 forward/backward and eval pass,
  quantizer projection has finite nonzero gradients; peak allocated GPU memory
  7.15 GiB for two synthetic 300-frame FBANK inputs (valid lengths 300 and 280).
  This smoke test does not run a full-size optimizer step or long-audio benchmark.

## Integration scope

Use `scripts/train.py` / `scripts/train_pretrain.sh` so `PretrainingTrainer` handles
summed losses correctly. Single-GPU PyTorch and DDP are supported; DeepSpeed and
FSDP are explicitly rejected pending sharded gradient normalization support.
Multi-rank normalization is implemented but not validated with a multi-GPU run.
Epoch lengths, saved steps and checkpoint intervals must align with gradient
accumulation boundaries.

The local optimizer, scheduler, batching and data recipe are not a reproduction of
fairseq2's complete pretraining recipe. Old CNN/SDConformer checkpoints do not match
the new defaults; use a fresh output directory. No pretrained checkpoint conversion
or convergence claim is made. Legacy CNN/SDConformer classes remain available for
explicit legacy configurations.
