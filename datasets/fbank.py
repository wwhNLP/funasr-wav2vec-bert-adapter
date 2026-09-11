"""CPU waveform conversion using the same Kaldi native FBANK backend as fairseq2."""
from __future__ import annotations

import torch


class WaveformToFbank:
    def __init__(self, sample_rate=16000, num_mel_bins=80, waveform_scale=1.0, standardize=False):
        import kaldi_native_fbank as knf
        self.knf = knf
        self.options = knf.FbankOptions()
        self.options.frame_opts.samp_freq = sample_rate
        self.options.mel_opts.num_bins = num_mel_bins
        self.waveform_scale = waveform_scale
        self.standardize = standardize

    def __call__(self, waveform):
        waveform = waveform.detach().cpu().float().reshape(-1) * self.waveform_scale
        computer = self.knf.OnlineFbank(self.options)
        computer.accept_waveform(self.options.frame_opts.samp_freq, waveform.tolist())
        computer.input_finished()
        if computer.num_frames_ready == 0:
            raise ValueError("Audio is too short to produce a FBANK frame.")
        import numpy as np
        features = torch.from_numpy(np.stack([computer.get_frame(i) for i in range(computer.num_frames_ready)]))
        if self.standardize:
            std, mean = torch.std_mean(features, dim=0)
            if not torch.isfinite(std).all() or (std == 0).any():
                raise ValueError("Cannot standardize FBANK with zero or non-finite variance.")
            features = (features - mean) / std
        return features
