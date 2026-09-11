# 适配审查与修复（2026-09-11）

使用 `codegraph init .` 初始化索引（初始 21 文件、297 节点、473 边），并用
`codegraph explore W2VBertModel` 检查模型调用关系。索引保留在本地 `.codegraph/`，不提交生成文件。

## 发现与处理

| 问题 | 修复 |
| --- | --- |
| 已提供 `model_conf` 时，FunASR 跳过远程加载；launcher 又把模型类名覆盖成目录 | 新增 `scripts/train.py`，先注册再进入真实 FunASR Hydra 训练入口；保留 YAML 类名 |
| `remote_code=./` 依赖工作目录，目录也不符合动态导入文件协议；绝对顶层导入容易冲突 | 新增 `register.py` 及独立包命名空间，内部使用相对导入 |
| Wav2Vec2 返回 dict，Trainer 解包要求 `(loss, stats, weight)` | 对齐训练接口并 detach 日志指标 |
| `eval()` 关闭掩码，预训练验证直接抛 `temporal_mask is None` | 验证保留预训练目标需要的掩码；关闭的 dropout 与量化采样仍遵循 eval 行为 |
| 掩码原地修改可能破坏无投影分支的反向传播；极短音频负采样无效 | 非原地掩码；不足两个 masked frames 时给出明确错误 |
| Conformer mask 按最大有效长度生成，不能处理额外右侧 padding | 按输入张量时间维构造 mask |
| BERT 非有限 loss 报错依赖已初始化的分布式组 | 直接检测非有限 loss；提前校验 BERT 层数与目标码本数 |
| 内嵌 DataPipe 未配置 rank/worker 分片，重复消费音频 | 在 IterableDataset 内显式按 rank/worker 分片，epoch 使用固定局部随机种子 |
| Trainer 实际调用 `batch_sampler.set_epoch`，原 Iterable loader 不支持；恢复忽略 start_step；各 rank 步数可不同 | 包装 epoch loader，提供接口、精确步数、循环补足及恢复跳过逻辑 |
| torchaudio 缺 TorchCodec 时吞掉解码异常，最终零批次 | SoundFile 回退及带样本 key 的错误日志；空 rank 明确报错 |
| 未重采样；tar 不同目录相同 basename 的音频互相覆盖 | 重采样后做长度过滤；使用完整 tar member 路径作为 key |
| 配置引用不存在的 `WarmupPolynomialDecayLR` | 实现并注册 warmup + polynomial decay，验证端点和状态恢复 |
| 配置混入未实现的 fbank/增强参数、空 tokenizer、过大的排序缓冲 | 清理无效设置，明确 raw-waveform CNN 路径，降低缓冲和预取量 |
| FunASR 默认按 acc 排名，自监督模型 checkpoint 被排除 | 显式设置 `avg_keep_nbest_models_type: loss` |

## 验证

环境：FunASR 1.4.15、PyTorch/torchaudio 2.11.0+cu128，GPU 4 NVIDIA A800 80GB。
依赖装在独立临时环境 `/tmp/w2vbert-review-env`，未修改已有 Python 环境。

```bash
CUDA_VISIBLE_DEVICES=4 /tmp/w2vbert-review-env/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=4 /tmp/w2vbert-review-env/bin/python scripts/smoke_forward.py --device cuda
```

最终 8 项回归全部通过（114.3 秒）。回归覆盖两个模型的 FP32/BF16 多步反向及验证、量化器有限非零梯度、额外 padding、
无投影前端、短音频错误、仓库外远程导入、多 worker/rank 分片、8k→16k 重采样、
单/多 worker 数据顺序恢复、调度器端点和状态恢复、真实 launcher 的训练/验证/checkpoint 恢复。

原始主配置模型也在 GPU 4 完成 BF16 前向、反向、AdamW 更新和 eval：
623,636,992 参数，输入两条 48,000-sample 波形（有效长度 48,000/45,000）；
训练 loss 1683.67，验证 loss 1416.42，峰值分配显存约 10.19 GiB。
以上 loss 用于有限性检查，不代表识别质量。

## 验证边界与后续工作

- 使用合成音频，未执行真实数据长期预训练或收敛评估。
- rank 分片采用两个逻辑 rank 加真实多 worker 验证；未占用其他卡进行多 GPU DDP/DeepSpeed 联调。
- `batch_num_epoch` 现在是精确的每 rank 步数，不再仅作估算；短数据会循环。
- 保持 rank 数、worker 数、分片列表和 batching 参数不变才能恢复相同数据顺序。
- 量化器现有温度计数仍按训练 forward 次数递增，梯度累积下不等于优化器更新次数。
- 当前实现是 raw-waveform CNN 自监督模型，不是原始 FBANK w2v-BERT 的逐层等价实现；
  未实现任意预训练权重转换或下游 ASR fine-tuning。审查修复聚焦随仓库提供的自监督 tar 训练路径。
