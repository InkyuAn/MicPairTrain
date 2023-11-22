import math
from typing import Optional, Tuple

import torch
from torch import nn, einsum
import warnings

from torch import Tensor
import torchaudio

class learnable_mel_scale_filter(nn.Module):
    def __init__(
            self,
            *,
            n_freq,
            n_ch,
            fs,
            n_mels,
            n_filter=16
    ):
        super().__init__()

        # mel_fb = torchaudio.functional.create_fb_matrix(n_freq, 0, fs // 2, n_mels, fs)
        mel_fb = self.create_fb_matrix(n_freq, 0, fs // 2, n_mels, fs)

        self._indice_mel_fb = mel_fb > 0
        self._n_filter = n_filter
        self._n_mels = n_mels

        self._learnable_filter = nn.ModuleList()
        for mel_idx in range(n_mels):
            mel_filter_size = int(torch.sum(self._indice_mel_fb[:, mel_idx]))
            tmp_mel_filter = nn.Sequential(
                nn.Conv1d(n_ch, n_filter, mel_filter_size, stride=mel_filter_size),
                nn.ReLU()
            )
            self._learnable_filter.append(tmp_mel_filter)

    def forward(self, x):
        # print("Debugging")
        b, c, n_freq, n_time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(-1, c, n_freq)

        # out_x = torch.zeros((b*n_time, self._n_filter, self._n_mels))
        out_x = []
        for idx, tmp_mel_filter in enumerate(self._learnable_filter):

            # out_x[:, :, idx] = tmp_mel_filter(x[:, :, self._indice_mel_fb[:, idx]])
            tmp_out_x = tmp_mel_filter(x[:, :, self._indice_mel_fb[:, idx]])
            # out_x[:, :, idx] = tmp_out_x.squeeze(2)
            out_x.append(tmp_out_x.squeeze(2))
        out_x = torch.stack(out_x, dim=-1)
        out_x = out_x.reshape(b, n_time, self._n_filter, self._n_mels)
        out_x = out_x.permute(0, 2, 3, 1)
        return out_x

    def create_fb_matrix( self,
            n_freqs: int,
            f_min: float,
            f_max: float,
            n_mels: int,
            sample_rate: int,
            norm: Optional[str] = None
    ) -> Tensor:
        r"""Create a frequency bin conversion matrix.

        Args:
            n_freqs (int): Number of frequencies to highlight/apply
            f_min (float): Minimum frequency (Hz)
            f_max (float): Maximum frequency (Hz)
            n_mels (int): Number of mel filterbanks
            sample_rate (int): Sample rate of the audio waveform
            norm (Optional[str]): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)

        Returns:
            Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
            meaning number of frequencies to highlight/apply to x the number of filterbanks.
            Each column is a filterbank so that assuming there is a matrix A of
            size (..., ``n_freqs``), the applied result would be
            ``A * create_fb_matrix(A.size(-1), ...)``.
        """

        if norm is not None and norm != "slaney":
            raise ValueError("norm must be one of None or 'slaney'")

        # freq bins
        # Equivalent filterbank construction by Librosa
        all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

        # calculate mel freq bins
        # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
        m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
        m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
        m_pts = torch.linspace(m_min, m_max, n_mels + 2)
        # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
        f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
        # calculate the difference between each mel point and each stft freq point in hertz
        f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
        # create overlapping triangles
        zero = torch.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
        up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
        fb = torch.max(zero, torch.min(down_slopes, up_slopes))

        if norm is not None and norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
            fb *= enorm.unsqueeze(0)

        if (fb.max(dim=0).values == 0.).any():
            warnings.warn(
                "At least one mel filterbank has all zero values. "
                f"The value for `n_mels` ({n_mels}) may be set too high. "
                f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
            )

        return fb