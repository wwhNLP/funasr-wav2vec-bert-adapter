import torch
import logging
import traceback
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "SelfSupervisedAudioDataset")
class SelfSupervisedAudioDataset(torch.utils.data.Dataset):
    """
    自监督音频数据集
    只返回音频特征，不包含文本标签，用于自监督学习
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        is_training: bool = True,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        retry: int = 5,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)

        self.preprocessor_speech = None
        if is_training:
            preprocessor_speech = kwargs.get("preprocessor_speech", None)
            if preprocessor_speech:
                preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
                preprocessor_speech = preprocessor_speech_class(
                    **kwargs.get("preprocessor_speech_conf")
                )
            self.preprocessor_speech = preprocessor_speech

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.retry = retry

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        """
        只返回音频特征，不包含文本标签，用于自监督学习任务
        """
        output = None
        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]
            source = item["source"]
            
            try:
                data_src = load_audio_text_image_video(source, fs=self.fs)
            except Exception as e:
                logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")
                continue

            if self.preprocessor_speech:
                data_src = self.preprocessor_speech(data_src, fs=self.fs)

            speech = data_src
            if speech.shape[0] == 0:
                logging.warning(f"Found empty audio file: {source}, skipping.")
                return None
            speech_lengths = torch.tensor(speech.shape[0], dtype=torch.int32)

            # 只返回音频特征，不包含文本
            output = {
                "speech": speech,
                "speech_lengths": speech_lengths,
            }
            break
        return output

    def collator(self, samples: list = None):
        """
        自监督学习的collator，只处理音频数据
        """
        original_size = len(samples)
        valid_samples = [s for s in samples if s is not None and s.get("speech") is not None and s["speech"].numel() > 0]
        
        if len(valid_samples) < original_size:
            logging.debug(f"Filtered out {original_size - len(valid_samples)} empty/invalid samples from a batch.")

        if not valid_samples:
            logging.warning("All samples in a batch were invalid, returning a dummy batch.")
            return {
                "speech": torch.rand(1, 16000, dtype=torch.float32),
                "speech_lengths": torch.tensor([16000], dtype=torch.int32),
            }

        outputs = {}
        for sample in valid_samples:
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        # 对音频特征进行padding
        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                # For sequence data (dim > 0), pad them.
                if data_list[0].dim() > 0:
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
                # For scalar data (dim = 0), stack them.
                else:
                    outputs[key] = torch.stack(data_list)

        return outputs


@tables.register("dataset_classes", "SelfSupervisedAudioDatasetWithAugmentation")
class SelfSupervisedAudioDatasetWithAugmentation(torch.utils.data.Dataset):
    """
    带数据增强的自监督音频数据集，支持对比学习等自监督学习任务
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        is_training: bool = True,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        retry: int = 5,
        # 数据增强参数
        noise_std: float = 0.01,
        time_stretch_range: list = [0.9, 1.1],
        pitch_shift_range: list = [-2, 2],
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)

        self.preprocessor_speech = None
        if is_training:
            preprocessor_speech = kwargs.get("preprocessor_speech", None)
            if preprocessor_speech:
                preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
                preprocessor_speech = preprocessor_speech_class(
                    **kwargs.get("preprocessor_speech_conf")
                )
            self.preprocessor_speech = preprocessor_speech

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.retry = retry
        
        # 数据增强参数
        self.noise_std = noise_std
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def __len__(self):
        return len(self.index_ds)

    def _add_noise(self, speech):
        """添加高斯噪声"""
        noise = torch.randn_like(speech) * self.noise_std
        return speech + noise

    def _time_stretch(self, speech, speech_lengths):
        """时间拉伸"""
        stretch_factor = torch.rand(1).mul(
            self.time_stretch_range[1] - self.time_stretch_range[0]
        ).add(self.time_stretch_range[0]).item()
        
        # 简单的线性插值实现时间拉伸
        original_length = speech.size(0)
        new_length = int(original_length * stretch_factor)
        
        if new_length > 0 and new_length != original_length:
            # 使用torch的插值功能
            # Reshape for 1D interpolation: (T) -> (1, 1, T)
            speech_for_interp = speech.view(1, 1, -1)
            interpolated_speech = torch.nn.functional.interpolate(
                speech_for_interp,
                size=new_length,
                mode='linear',
                align_corners=False
            )
            # Reshape back to (T): (1, 1, new_T) -> (new_T)
            speech = interpolated_speech.view(-1)
            
            speech_lengths = torch.tensor(new_length, dtype=torch.int32)
        
        return speech, speech_lengths

    def __getitem__(self, index):
        """
        返回原始音频和增强后的音频，用于对比学习
        """
        output = None
        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]
            source = item["source"]
            
            try:
                data_src = load_audio_text_image_video(source, fs=self.fs)
            except Exception as e:
                logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")
                continue

            if self.preprocessor_speech:
                data_src = self.preprocessor_speech(data_src, fs=self.fs)

            speech = data_src
            if speech.shape[0] == 0:
                logging.warning(f"Found empty audio file: {source}, skipping.")
                return None
            speech_lengths = torch.tensor(speech.shape[0], dtype=torch.int32)

            # 创建两个不同的增强版本用于对比学习
            speech_aug1 = self._add_noise(speech)
            speech_aug2, speech_lengths_aug2 = self._time_stretch(speech, speech_lengths)
            speech_aug2 = self._add_noise(speech_aug2)

            output = {
                "speech": speech,  # 原始音频
                "speech_lengths": speech_lengths,
                "speech_aug1": speech_aug1,  # 增强版本1
                "speech_aug2": speech_aug2,  # 增强版本2
                "speech_lengths_aug2": speech_lengths_aug2,
            }
            break

        return output

    def collator(self, samples: list = None):
        """
        带增强的自监督学习collator
        """
        original_size = len(samples)
        valid_samples = [s for s in samples if s is not None and s.get("speech") is not None and s["speech"].numel() > 0]
        
        if len(valid_samples) < original_size:
            logging.debug(f"Filtered out {original_size - len(valid_samples)} empty/invalid samples from a batch.")

        if not valid_samples:
            logging.warning("All samples in a batch were invalid, returning a dummy batch.")
            return {
                "speech": torch.rand(1, 16000, dtype=torch.float32),
                "speech_lengths": torch.tensor([16000], dtype=torch.int32),
                "speech_aug1": torch.rand(1, 16000, dtype=torch.float32),
                "speech_aug2": torch.rand(1, 16000, dtype=torch.float32),
                "speech_lengths_aug2": torch.tensor([16000], dtype=torch.int32),
            }
            
        outputs = {}
        for sample in valid_samples:
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        # 对所有音频特征进行padding
        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                # For sequence data (dim > 0), pad them.
                if data_list[0].dim() > 0:
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
                # For scalar data (dim = 0), stack them.
                else:
                    outputs[key] = torch.stack(data_list)

        return outputs
