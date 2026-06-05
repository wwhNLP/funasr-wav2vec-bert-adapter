
# WENET/TorchAudio-style DataPipe for shard reading
# Copied and adapted from WeNet codebase
# https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/datapipes.py
import tarfile
import io
import os
import torchaudio
from torch.utils.data import IterableDataset, IterDataPipe, functional_datapipe
import torch
import random
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

def read_tar_file(data):
    """Read a tar file and yield samples.
    Args:
        data: A dict containing webdataset path.
    Returns:
        A list of samples. Each sample is a dict.
    """
    path = data['path']
    samples = []
    obj = {}
    try:
        with tarfile.open(path, "r:*") as tar:
            for member in tar.getmembers():
                key = member.name
                key = key.replace('\\', '/')
                key = os.path.splitext(key.split('/')[-1])[0]
                if key not in obj:
                    obj[key] = {}

                fp = tar.extractfile(member)
                if fp is None:
                    continue
                content = fp.read()

                ext = os.path.splitext(member.name)[1]
                if ext == '.txt':
                    obj[key]['txt'] = content
                elif ext[1:] in AUDIO_FORMAT_SETS:
                    obj[key]['wav'] = content

    except Exception as e:
        # empty tar, corrupted tar, unfinished tar......直接 skip
        print(f"[WARNING] skip bad tar: {path}, reason: {e}")
        return samples   # empty list

    for key, value in obj.items():
        # For self-supervised learning, we only need the audio.
        if 'wav' in value:
            sample = dict(key=key, wav=value['wav'])
            # Add a placeholder for 'txt' if it's missing, to avoid downstream errors.
            if 'txt' not in value:
                sample['txt'] = b'' # empty bytes
            samples.append(sample)

    return samples

class Shuffler:
    """A simple shuffler for iter datapipe."""
    def __init__(self, shuffler_size=1000, shuffle=True):
        self.shuffler_size = shuffler_size
        self.shuffle = shuffle
        self.pool = []

    def add(self, x):
        """Add a sample to the pool."""
        self.pool.append(x)
        if len(self.pool) >= self.shuffler_size:
            if self.shuffle:
                idx = random.randint(0, len(self.pool) - 1)
                return self.pool.pop(idx)
            else:
                return self.pool.pop(0)
        else:
            return None

    def get_and_clear(self):
        """Get all samples from the pool and clear it."""
        if self.shuffle:
            random.shuffle(self.pool)
        for x in self.pool:
            yield x
        self.pool = []

# @functional_datapipe("tar_file_and_group")
class TarFileAndGroupDataPipe(IterDataPipe):
    """
    This DataPipe reads samples from tar files and applies streaming shuffling.
    The `read_tar_file` function is expected to group samples by key within a single tar file.
    """
    def __init__(self, datapipe, shuffle=False, shuffler_size=1000):
        self.datapipe = datapipe
        self.shuffle = shuffle
        self.shuffler_size = shuffler_size

    def __iter__(self):
        shuffler = Shuffler(self.shuffler_size, self.shuffle)
        for data in self.datapipe:
            # 读取一个tar的样本数据
            samples = read_tar_file(data)
            for sample in samples:
                # 添加到shuffler中
                shuffled_sample = shuffler.add(sample)
                if shuffled_sample is not None:
                    yield shuffled_sample
        
        # Yield remaining items in the shuffler
        for sample in shuffler.get_and_clear():
            yield sample

try:
    if not hasattr(IterDataPipe, "tar_file_and_group"):
        IterDataPipe.register_datapipe_as_function("tar_file_and_group", TarFileAndGroupDataPipe)
except Exception as e:
    print(f"[WARNING] Failed to register tar_file_and_group: {e}")




# @functional_datapipe("dynamic_bucket_batch")
class DynamicBucketBatcher(IterDataPipe):
    def __init__(self, datapipe, **kwargs):
        self.datapipe = datapipe
        self.kwargs = kwargs
        self.batch_size = self.kwargs.get("batch_size", 16)
        self.sort_size = self.kwargs.get("sort_size", 1000)
        self.batch_type = self.kwargs.get("batch_type", "token")

    def _create_batches(self, buffer):
        # Sort the buffer by speech length
        sorted_buffer = sorted(buffer, key=lambda x: x["speech_lengths"].item())
        batch = []
        max_len_in_batch = 0
        for item in sorted_buffer:
            if self.batch_type == "example":
                current_len = 1
            else:  # token/frame
                current_len = item["speech_lengths"].item()

            potential_batch_len = max(max_len_in_batch, current_len) * (len(batch) + 1)
            if potential_batch_len <= self.batch_size:
                batch.append(item)
                max_len_in_batch = max(max_len_in_batch, current_len)
            else:
                if batch:
                    yield batch
                batch = [item]
                max_len_in_batch = current_len
        if batch:
            yield batch

    def __iter__(self):
        buffer = []
        for sample in self.datapipe:
            buffer.append(sample)
            if len(buffer) >= self.sort_size:
                yield from self._create_batches(buffer)
                buffer = []

        if buffer:
            yield from self._create_batches(buffer)

try:
    if not hasattr(IterDataPipe, "dynamic_bucket_batch"):
        IterDataPipe.register_datapipe_as_function("dynamic_bucket_batch", DynamicBucketBatcher)
except Exception as e:
    print(f"[WARNING] Failed to register dynamic_bucket_batch: {e}")


@tables.register("dataset_classes", "LargeAudioDataset")
class LargeAudioDataset(IterableDataset):
    """
    LargeDataset for reading from sharded tar files.
    """
    def __init__(
        self,
        path,
        frontend=None,
        tokenizer=None,
        is_training: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.kwargs = kwargs

        self.preprocessor_speech = None
        self.preprocessor_text = None

        if is_training:
            preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf", {})
            preprocessor_speech = kwargs.get("preprocessor_speech", None)
            if preprocessor_speech:
                preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
                self.preprocessor_speech = preprocessor_speech_class(**preprocessor_speech_conf)

            preprocessor_text_conf = kwargs.get("preprocessor_text_conf", {})
            preprocessor_text = kwargs.get("preprocessor_text", None)
            if preprocessor_text:
                preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
                self.preprocessor_text = preprocessor_text_class(**preprocessor_text_conf)

        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.int_pad_value = kwargs.get("int_pad_value", -1)
        self.float_pad_value = kwargs.get("float_pad_value", 0.0)
        self.min_wav_len = kwargs.get("min_wav_len", 1)
        self.max_wav_len = kwargs.get("max_wav_len", float('inf'))
        self.prompt_prefix = kwargs.get("prompt_prefix", "")

        with open(path, 'r', encoding='utf8') as fin:
            self.shard_list = [line.strip() for line in fin]

    def _process_sample(self, sample: dict) -> dict:
        wav_bytes = sample['wav']
        target = sample['txt']

        wav_file = io.BytesIO(wav_bytes)
        data_src, fs = torchaudio.load(wav_file)
        
        if data_src.shape[1] < self.min_wav_len or data_src.shape[1] > self.max_wav_len:
            return None

        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=fs)

        speech, speech_lengths = extract_fbank(
            data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
        )

        if self.preprocessor_text:
            target = self.preprocessor_text(target)
        
        # Add prefix if exists (e.g. "<|ja|><|NEUTRAL|><|Speech|><|woitn|>")
        if self.prompt_prefix:
            if isinstance(target, bytes):
                target = target.decode("utf-8")
            target = self.prompt_prefix + target

        if self.tokenizer:
            ids = self.tokenizer.encode(target)
            text = torch.tensor(ids, dtype=torch.int64)
        else:
            ids = target
            text = torch.tensor(ids, dtype=torch.int64)

        ids_lengths = len(ids)
        text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)
        
        return {
            "speech": speech[0, :, :],
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
        }

    def collator(self, samples: list) -> dict:
        outputs = {}
        for sample in samples:
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        for key, data_list in outputs.items():
            if data_list and isinstance(data_list[0], torch.Tensor):
                # --- Pre-emptive Debugging ---
                # This will print details for every batch before padding.
                # The last printout before the crash will be the problematic batch.
                print(f"\n[DEBUG] Pre-padding check for key='{key}'. Batch size: {len(data_list)}", flush=True)
                for i, t in enumerate(data_list):
                    print(f"  - Tensor {i}: shape={t.shape}, dtype={t.dtype}, numel: {t.numel()}", flush=True)
                # --- End Debugging ---

                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                    pad_value = self.int_pad_value
                else:
                    pad_value = self.float_pad_value

                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )
        return outputs

    def __iter__(self):
        # tar的datapipe
        datapipe = torch.utils.data.datapipes.iter.IterableWrapper(self.shard_list)
        # 多GPU分发
        datapipe = datapipe.sharding_filter()
        # 训练时shuffle
        if self.is_training:
            datapipe = datapipe.shuffle()
        # 将路径转换为字典
        datapipe = datapipe.map(lambda x: dict(path=x))
        shuffler_size = self.kwargs.get("shuffler_size", 2000)
        datapipe = datapipe.tar_file_and_group(shuffle=self.is_training, shuffler_size=shuffler_size)
        # process sample
        datapipe = datapipe.map(self._process_sample)
        # 过滤上步骤返回None的样本
        datapipe = datapipe.filter(lambda x: x is not None)
        # dynamic batching
        datapipe = datapipe.dynamic_bucket_batch(**self.kwargs)
        # 合并样本
        datapipe = datapipe.map(self.collator)

        return iter(datapipe)



import torch
import torchaudio
import json
import os
import io
import logging
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter import IterableWrapper

@tables.register("dataset_classes", "SupervisedLargeAudioDataset")
class SupervisedLargeAudioDataset(IterableDataset):
    """
    [终极兼容版] 
    1. 支持 Tar Shards (原版功能)
    2. 支持 JSONL + 本地 Wav 路径 (新功能)
    3. 自动识别键名 (source/wav, target/txt)
    """
    def __init__(
        self,
        path,
        frontend=None,
        tokenizer=None,
        is_training: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        
        # 🔥🔥🔥 调试打印：看看系统到底读了哪个文件？
        print(f"\n[Debug] Data List Path: {self.path}")
        # 🔥🔥🔥
        
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.kwargs = kwargs

        self.preprocessor_speech = None
        self.preprocessor_text = None

        # === FIX: Force use HF AutoTokenizer for Seamless ===
        # FunASR SentencepiecesTokenizer may mismatch with Seamless BPE
        try:
            from transformers import AutoTokenizer
            # Hardcoded path for stability during debug
            hf_path = "/workspace/wyh/Wav2vec_and_senseVoice/models/seamless-m4t-v2-large"
            print(f"[Dataset] Force loading HF AutoTokenizer from: {hf_path}")
            # 尝试指定语言为中文，防止默认变成法语
            self.tokenizer = AutoTokenizer.from_pretrained(hf_path, src_lang="cmn", tgt_lang="cmn")
            print("[Dataset] HF Tokenizer loaded successfully.")
        except Exception as e:
            print(f"[Dataset] Failed to load HF Tokenizer: {e}. Fallback to default.")
        # ====================================================

        if is_training:
            self._init_preprocessor("preprocessor_speech", kwargs)
            self._init_preprocessor("preprocessor_text", kwargs)

        self.fs = 16000 if frontend is None else frontend.fs
        self.int_pad_value = kwargs.get("int_pad_value", -1)
        self.float_pad_value = kwargs.get("float_pad_value", 0.0)
        self.min_wav_len = kwargs.get("min_wav_len", 3200)
        self.max_wav_len = kwargs.get("max_wav_len", 480000)

        with open(path, 'r', encoding='utf8') as fin:
            self.shard_list = [line.strip() for line in fin]
        
        # --- 改进的判断逻辑 ---
        # 不再只看文件名，而是看第一行的内容
        # 如果第一行是以 "{" 开头，那绝对是 JSONL，不管路径里有没有 .tar
        first_line = self.shard_list[0] if len(self.shard_list) > 0 else ""
        
        if first_line.strip().startswith("{"):
            self.is_tar_mode = False
            logging.info(f"Dataset detected JSONL mode (starts with '{{'). Loaded {len(self.shard_list)} samples.")
        else:
            self.is_tar_mode = True
            logging.info(f"Dataset detected TAR shards mode (default). Found {len(self.shard_list)} files.")

        if not self.is_tar_mode:
            self.json_data = []
            for line in self.shard_list:
                try:
                    self.json_data.append(json.loads(line))
                except:
                    pass
    def __len__(self):
        """
        为 Trainer 提供数据集长度防止报错
        """
        # 1. 如果是 JSONL 模式，直接返回数据行数
        if hasattr(self, 'json_data') and self.json_data:
            return len(self.json_data)
        
        # 2. 如果是 Tar 模式，估算一个长度 (防止除零错误)
        if hasattr(self, 'shard_list'):
            # 假设每个 tar 包里有 1000 条数据（这只是为了防报错，数值不准没关系）
            return len(self.shard_list) * 1000
            
        return 0
    def _init_preprocessor(self, name, kwargs):
        conf = kwargs.get(f"{name}_conf", {})
        proc_name = kwargs.get(name, None)
        if proc_name:
            cls = tables.preprocessor_classes.get(proc_name)
            setattr(self, name, cls(**conf))

    def _process_sample(self, sample: dict) -> dict:
        # === DEBUG DATASET ===
        if not hasattr(self, "_sample_count"): self._sample_count = 0
        self._sample_count += 1
        verbose = self._sample_count <= 5 # 只打印前5个样本
        if verbose:
            print(f"\n[DEBUG DATASET SAMPLE {self._sample_count}]")
            print(f"  - Raw sample keys: {list(sample.keys())}")
            if 'path' in sample: print(f"  - Path: {sample['path']}")

        # 1. 统一键名 (source/wav -> wav_data, target/txt -> target)
        wav_data = sample.get('source') or sample.get('wav') or sample.get('audio')
        target = sample.get('target') or sample.get('txt') or sample.get('text') or b''
        
        if not wav_data:
            return None

        # 2. 处理音频加载 (核心兼容逻辑)
        try:
            if isinstance(wav_data, (bytes, bytearray)):
                # --- 情况A: 来自 Tar 包 (Bytes) ---
                wav_file = io.BytesIO(wav_data)
                data_src, fs = torchaudio.load(wav_file)
            elif isinstance(wav_data, str):
                # --- 情况B: 来自 JSONL (文件路径) ---
                if not os.path.exists(wav_data):
                    # 尝试相对路径拼接 (可选)
                    # wav_data = os.path.join(os.path.dirname(self.path), wav_data)
                    return None
                data_src, fs = torchaudio.load(wav_data)
            else:
                return None
        except Exception as e:
            # print(f"Error loading audio: {e}") 
            return None

        # 3. 长度过滤
        if data_src.shape[1] < self.min_wav_len or data_src.shape[1] > self.max_wav_len:
            return None

        # 4. 单通道与重采样
        if data_src.dim() > 1 and data_src.shape[0] > 1:
            data_src = torch.mean(data_src, dim=0, keepdim=True)
        
        if fs != self.fs:
            if not hasattr(self, "_resamplers"):
                self._resamplers = {}
            if fs not in self._resamplers:
                self._resamplers[fs] = torchaudio.transforms.Resample(fs, self.fs)
            data_src = self._resamplers[fs](data_src)

        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=self.fs)

        speech = data_src.squeeze(0).float()
        speech_lengths = torch.tensor([speech.shape[0]], dtype=torch.int32)

        # 5. 处理文本
        if isinstance(target, bytes):
            target = target.decode('utf-8').strip()
        
        if not target:
            return None
            
        # === Step 2: 修正中文空格 ===
        # 强制去除空格，避免模型学习大量空格 token
        target_no_space = target.replace(" ", "")
        
        # 为了对比，保留一份原始带空格的，但最终送入模型的是去空格的
        target_final = target_no_space
        # ==========================

        if self.tokenizer:
            # === DEEP DEBUG BLOCK (Step A) ===
            # 只在前几条样本打印，避免刷屏
            if verbose:
                print(f"\n[DATASET DEEP DEBUG]")
                print(f"1. Raw txt: '{target[:50]}...'")
                print(f"2. No-space txt: '{target_no_space[:50]}...'")
                
                # A. 测试原始带空格编码
                try:
                    # 强制不加特殊 token，只看文本本身
                    ids_raw = self.tokenizer.encode(target, add_special_tokens=False)
                    print(f"3. IDs (raw, no-special): {ids_raw[:20]}...")
                    if hasattr(self.tokenizer, "decode"):
                        dec_raw = self.tokenizer.decode(ids_raw)
                        print(f"4. Decode (raw): '{dec_raw[:50]}...'")
                except Exception as e:
                    print(f"   [Error encoding raw]: {e}")

                # B. 测试去空格编码 (实际使用的)
                try:
                    # 强制不加特殊 token
                    ids_final = self.tokenizer.encode(target_final, add_special_tokens=False)
                    print(f"5. IDs (final, no-special): {ids_final[:20]}...")
                    if hasattr(self.tokenizer, "decode"):
                        dec_final = self.tokenizer.decode(ids_final)
                        print(f"6. Decode (final): '{dec_final[:50]}...'")
                except Exception as e:
                    print(f"   [Error encoding final]: {e}")
                print("-" * 30 + "\n")
            # ===============================
        
        # === Step 1: DEBUG TOKENIZATION (Space vs No-Space) ===
        if verbose and self.tokenizer:
            # 强制不加特殊 token
            t_ids = self.tokenizer.encode(target_final, add_special_tokens=False)
            print(f"  - [TOKEN DEBUG] Processed txt tokens: {len(t_ids)}, content: '{target_final[:30]}...'")
        # =====================================================

        if self.preprocessor_text:
            target = self.preprocessor_text(target_final)
        else:
            target = target_final
        
        if self.tokenizer:
            # 最终输出给模型的 ID，绝对不能包含 special tokens，全交给模型层拼接
            ids = self.tokenizer.encode(target, add_special_tokens=False)
            text = torch.tensor(ids, dtype=torch.int64)
            if verbose:
                print(f"  - Target Text: {target[:50]}...")
                print(f"  - Encoded IDs (first 10): {ids[:10]}")
                # === Step 3: 标签自检 ===
                if hasattr(self.tokenizer, "decode"):
                    try:
                        decoded_text = self.tokenizer.decode(ids)
                        print(f"  - [SELF CHECK] Decode back: {decoded_text}")
                    except Exception as e:
                        print(f"  - [SELF CHECK] Decode failed: {e}")
                # =======================
        else:
            text = torch.tensor([], dtype=torch.int64)
            if verbose: print("  - WARNING: No tokenizer found in dataset!")

        text_lengths = torch.tensor([len(text)], dtype=torch.int32)
        
        return {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
        }

    def collator(self, samples: list) -> dict:
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        
        outputs = {}
        for sample in samples:
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        for key, data_list in outputs.items():
            if data_list and isinstance(data_list[0], torch.Tensor):
                padding_val = self.int_pad_value if data_list[0].dtype in (torch.int64, torch.int32) else self.float_pad_value
                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=padding_val
                )
        return outputs

    @staticmethod
    def _create_batches(buffer, batch_size, batch_type):
        # 排序与 Batch 逻辑 (保持不变)
        sorted_buffer = sorted(buffer, key=lambda x: x["speech_lengths"].item())
        batch = []
        max_len_in_batch = 0
        for item in sorted_buffer:
            current_len = item["speech_lengths"].item()
            if batch_type == "example":
                potential_len = len(batch) + 1
            else:
                max_len_in_batch = max(max_len_in_batch, current_len)
                potential_len = max_len_in_batch * (len(batch) + 1)

            if potential_len > batch_size:
                if batch: yield batch
                batch = [item]
                max_len_in_batch = current_len
            else:
                batch.append(item)
        if batch: yield batch

    def __iter__(self):
        # === 终极调试：打印所有的 kwargs 看看里面到底有什么 ===
        if not hasattr(self, "_printed_conf"):
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("\n" + "="*50)
                print("[DEBUG] Dataset self.kwargs keys:", list(self.kwargs.keys()))
                if "dataset_conf" in self.kwargs:
                    print("[DEBUG] dataset_conf content:", self.kwargs["dataset_conf"])
                print("="*50 + "\n")
            self._printed_conf = True

        # 尝试多路径读取 batch_size
        batch_size = 320000 # 默认值
        batch_type = "frame" # 默认值
        
        # 路径 1: 直接在 kwargs 下 ( Hydra ++batch_size )
        if "batch_size" in self.kwargs:
            batch_size = self.kwargs["batch_size"]
        if "batch_type" in self.kwargs:
            batch_type = self.kwargs["batch_type"]
            
        # 路径 2: 在 dataset_conf 下 ( Hydra ++dataset_conf.batch_size )
        if "dataset_conf" in self.kwargs:
            d_conf = self.kwargs["dataset_conf"]
            if hasattr(d_conf, "get"):
                batch_size = d_conf.get("batch_size", batch_size)
                batch_type = d_conf.get("batch_type", batch_type)
            else:
                # 处理 DictConfig 类型的特殊情况
                if "batch_size" in d_conf: batch_size = d_conf["batch_size"]
                if "batch_type" in d_conf: batch_type = d_conf["batch_type"]

        sort_size = self.kwargs.get("sort_size", 1000)
        
        # 打印最终生效值
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"\n>>> [FINAL CONFIG] Batch: {batch_size}, Type: {batch_type}, Sort: {sort_size}\n")

        if self.is_tar_mode:
            # === 模式 A: Tar Shards (原版逻辑) ===
            datapipe = IterableWrapper(self.shard_list)
            datapipe = datapipe.sharding_filter()
            if self.is_training:
                datapipe = datapipe.shuffle()
            datapipe = datapipe.map(lambda x: dict(path=x))
            shuffler_size = self.kwargs.get("shuffler_size", 2000)
            # 这里调用 tar 处理
            datapipe = datapipe.tar_file_and_group(shuffle=self.is_training, shuffler_size=shuffler_size)
        else:
            # === 模式 B: JSONL 文件 (新逻辑) ===
            datapipe = IterableWrapper(self.json_data)
            if self.is_training:
                datapipe = datapipe.shuffle()
        
        # 公共后续处理
        datapipe = datapipe.map(self._process_sample)
        datapipe = datapipe.filter(lambda x: x is not None)
        
        datapipe = datapipe.batch(sort_size)

        datapipe = datapipe.map(lambda buffer: list(self._create_batches(buffer, batch_size, batch_type)))
        datapipe = datapipe.unbatch()
        datapipe = datapipe.map(self.collator)
        datapipe = datapipe.filter(lambda x: x is not None)

        return iter(datapipe)





@tables.register("dataset_classes", "SelfSupervisedLargeAudioDataset")
class SelfSupervisedLargeAudioDataset(IterableDataset):
    """
    Unsupervised LargeDataset for reading from sharded tar files.
    """
    def __init__(
        self,
        path,
        frontend=None,
        is_training: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.frontend = frontend
        self.is_training = is_training
        self.kwargs = kwargs

        # --- START OF FIX B ---
        self.batch_num_epoch = self.kwargs.get("batch_num_epoch")
        if self.batch_num_epoch is None:
            raise ValueError(
                "Trainer requires a dataloader length. Please provide 'batch_num_epoch' "
                "(estimated total number of batches for one epoch) "
                "in the dataset configuration."
            )
        self.batch_num_epoch = int(self.batch_num_epoch)
        # --- END OF FIX B ---

        self.preprocessor_speech = None
        if is_training:
            preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf", {})
            preprocessor_speech = kwargs.get("preprocessor_speech", None)
            if preprocessor_speech:
                preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
                self.preprocessor_speech = preprocessor_speech_class(**preprocessor_speech_conf)

        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.int_pad_value = kwargs.get("int_pad_value", -1)
        self.float_pad_value = kwargs.get("float_pad_value", 0.0)
        self.min_wav_len = kwargs.get("min_wav_len", 1)
        self.max_wav_len = kwargs.get("max_wav_len", float('inf'))

        with open(path, 'r', encoding='utf8') as fin:
            self.shard_list = [line.strip() for line in fin]

    def __len__(self):
        """
        Returns the pre-estimated number of batches per epoch.
        This is required by the trainer to get a length for the dataloader.
        """
        return self.batch_num_epoch

    def _process_sample(self, sample: dict) -> dict:
        key = sample.get("key", "UNKNOWN_KEY")
        wav_bytes = sample['wav']

        if not wav_bytes:
            return None
        
        wav_file = io.BytesIO(wav_bytes)
        try:
            data_src, fs = torchaudio.load(wav_file)
        except Exception as e:
            return None

        if data_src.numel() == 0 or data_src.shape[1] < self.min_wav_len:
            return None

        if data_src.dim() > 1 and data_src.shape[0] > 1:
            data_src = torch.mean(data_src, dim=0, keepdim=True)

        if data_src.shape[1] > self.max_wav_len:
            return None

        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=fs)

        return {
            "speech": data_src.squeeze(0),
            "speech_lengths": data_src.shape[1],  # Return as integer
        }

    def collator(self, batch_of_samples: list) -> dict:
        """
        This collator receives a small batch of samples, pads sequences and stacks scalars.
        """
        if not batch_of_samples:
            return None

        outputs = {}
        # Group samples by key
        for sample in batch_of_samples:
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])
        
        if not outputs:
            return None

        # Convert list of lengths to a tensor
        if 'speech_lengths' in outputs:
            outputs['speech_lengths'] = torch.tensor(outputs['speech_lengths'], dtype=torch.int32)

        # Pad other sequences which are tensors
        for key, data_list in outputs.items():
            if key == 'speech_lengths':
                continue # Already handled

            if data_list and isinstance(data_list[0], torch.Tensor):
                pad_value = self.float_pad_value
                if data_list[0].dtype in (torch.int64, torch.int32):
                    pad_value = self.int_pad_value
                
                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )
        return outputs

    @staticmethod
    def _create_batches(buffer, batch_size, batch_type):
        # Sort the buffer by speech length
        sorted_buffer = sorted(buffer, key=lambda x: x["speech_lengths"])
        
        batch = []
        max_len_in_batch = 0
        for item in sorted_buffer:
            current_len = item["speech_lengths"]
            if batch_type == "example":
                potential_batch_len = (len(batch) + 1)
            else:  # frame or token
                max_len_in_batch = max(max_len_in_batch, current_len)
                potential_batch_len = max_len_in_batch * (len(batch) + 1)

            if potential_batch_len > batch_size:
                if batch:
                    yield batch
                batch = [item]
                max_len_in_batch = current_len
            else:
                batch.append(item)

        if batch:
            yield batch

    def __iter__(self):
        datapipe = torch.utils.data.datapipes.iter.IterableWrapper(self.shard_list)
        datapipe = datapipe.sharding_filter()
        if self.is_training:
            datapipe = datapipe.shuffle()
        datapipe = datapipe.map(lambda x: dict(path=x))
        shuffler_size = self.kwargs.get("shuffler_size", 2000)
        datapipe = datapipe.tar_file_and_group(shuffle=self.is_training, shuffler_size=shuffler_size)
        datapipe = datapipe.map(self._process_sample)
        datapipe = datapipe.filter(lambda x: x is not None)
        
        # Buffer samples for sorting
        sort_size = self.kwargs.get("sort_size", 5000)
        datapipe = datapipe.batch(sort_size)

        # Create dynamic mini-batches from the sorted buffer
        batch_size = self.kwargs.get("batch_size", 500000)
        batch_type = self.kwargs.get("batch_type", "frame")
        
        # Use map to convert the generator from _create_batches into a list of batches
        datapipe = datapipe.map(
            lambda buffer: list(self._create_batches(buffer, batch_size, batch_type))
        )
        
        # Unbatch the list of batches into a stream of batches
        datapipe = datapipe.unbatch()

        # Collate each mini-batch
        datapipe = datapipe.map(self.collator)
        datapipe = datapipe.filter(lambda x: x is not None)

        return iter(datapipe)


@tables.register("dataset_classes", "SelfSupervisedLargeAudioDatasetWithAugmentation")
class SelfSupervisedLargeAudioDatasetWithAugmentation(SelfSupervisedLargeAudioDataset):
    """
    Unsupervised LargeDataset with data augmentation for contrastive learning.
    """
    def __init__(
        self,
        *args,
        noise_std: float = 0.01,
        time_stretch_range: list = [0.9, 1.1],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Data augmentation parameters
        self.noise_std = noise_std
        self.time_stretch_range = time_stretch_range

    def _add_noise(self, speech):
        """Adds Gaussian noise."""
        noise = torch.randn_like(speech) * self.noise_std
        return speech + noise

    def _time_stretch(self, speech, speech_lengths):
        """Time stretch."""
        stretch_factor = torch.rand(1).mul(
            self.time_stretch_range[1] - self.time_stretch_range[0]
        ).add(self.time_stretch_range[0]).item()
        
        original_length = speech.size(0)
        new_length = int(original_length * stretch_factor)
        
        if new_length > 0 and new_length != original_length:
            speech_for_interp = speech.transpose(0, 1).unsqueeze(0)
            interpolated_speech = torch.nn.functional.interpolate(
                speech_for_interp,
                size=new_length,
                mode='linear',
                align_corners=False
            )
            speech = interpolated_speech.squeeze(0).transpose(0, 1)
            speech_lengths = torch.tensor([new_length], dtype=torch.int32)
        
        return speech, speech_lengths

    def _process_sample(self, sample: dict) -> dict:
        wav_bytes = sample['wav']
        
        wav_file = io.BytesIO(wav_bytes)
        data_src, fs = torchaudio.load(wav_file)

        # Ensure audio is mono
        if data_src.dim() > 1 and data_src.shape[0] > 1:
            data_src = torch.mean(data_src, dim=0, keepdim=True)

        if data_src.shape[1] < self.min_wav_len or data_src.shape[1] > self.max_wav_len:
            return None

        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=fs)

        speech, speech_lengths = extract_fbank(
            data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
        )
        speech = speech[0, :, :]

        # Create two different augmented views for contrastive learning
        speech_aug1 = self._add_noise(speech)
        speech_aug2, speech_lengths_aug2 = self._time_stretch(speech, speech_lengths)
        speech_aug2 = self._add_noise(speech_aug2)

        return {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "speech_aug1": speech_aug1,
            "speech_aug2": speech_aug2,
            "speech_lengths_aug2": speech_lengths_aug2,
        }
