import librosa
import numpy as np
import math

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

### IK, 20221124, Backup
# def compute_stft(model_version, audios, audio_ch, _fr_size, _hop_label_len,
#                  _stft_nfft, _stft_hop_len, stft_win_len):
#     frame_size = _fr_size
#     hop_label_len = _hop_label_len
#     audio_stft = []
#
#     if model_version == 1: # W/O DC
#         for idxfr in range(frame_size):
#             audio_ft_stft = []
#             audios_fr = audios[hop_label_len * idxfr: hop_label_len * (idxfr + 1), :]
#
#             for micIdx in range(audio_ch):
#                 mic_audio_fr_stft = librosa.stft(audios_fr[:, micIdx],
#                                                  n_fft=_stft_nfft,
#                                                  hop_length=_stft_hop_len,
#                                                  win_length=stft_win_len,
#                                                  center=True
#                                                  )
#                 audio_ft_stft.append(mic_audio_fr_stft[1:, :])  # W/o DC
#             audio_stft.append(np.stack(audio_ft_stft, axis=0))
#         audio_stft = np.concatenate(audio_stft, axis=-1)
#     elif model_version == 2: # W/ DC
#         for idxfr in range(frame_size):
#             audio_ft_stft = []
#             audios_fr = audios[hop_label_len * idxfr: hop_label_len * (idxfr + 1), :]
#
#             for micIdx in range(audio_ch):
#                 mic_audio_fr_stft = librosa.stft(audios_fr[:, micIdx],
#                                                  n_fft=_stft_nfft,
#                                                  hop_length=_stft_hop_len,
#                                                  win_length=stft_win_len,
#                                                  center=True
#                                                  )
#                 audio_ft_stft.append(mic_audio_fr_stft)
#             audio_stft.append(np.stack(audio_ft_stft, axis=0))
#         audio_stft = np.concatenate(audio_stft, axis=-1)
#     return audio_stft

### IK, 20221124, modify computing a STFT signal
def compute_stft(model_version, audios, audio_ch, _fr_size, _hop_label_len,
                 _stft_nfft, _stft_hop_len, stft_win_len):
    frame_size = _fr_size
    hop_label_len = _hop_label_len
    audio_stft = []

    if model_version == 1: # W/O DC
        audio_stft = []
        for micIdx in range(audio_ch):
            mic_audio_stft = librosa.stft(audios[:, micIdx],
                                          n_fft=_stft_nfft,
                                          hop_length=_stft_hop_len,
                                          win_length=stft_win_len,
                                          # center=True
                                          center=True
                                          )
            audio_stft.append(mic_audio_stft[1:, 1:])    # W/o DC
        audio_stft = np.stack(audio_stft, axis=0)
        # for idxfr in range(frame_size):
        #     audio_ft_stft = []
        #     audios_fr = audios[hop_label_len * idxfr: hop_label_len * (idxfr + 1), :]
        #
        #     for micIdx in range(audio_ch):
        #         mic_audio_fr_stft = librosa.stft(audios_fr[:, micIdx],
        #                                          n_fft=_stft_nfft,
        #                                          hop_length=_stft_hop_len,
        #                                          win_length=stft_win_len,
        #                                          center=True
        #                                          )
        #         audio_ft_stft.append(mic_audio_fr_stft[1:, :])  # W/o DC
        #     audio_stft.append(np.stack(audio_ft_stft, axis=0))
        # audio_stft = np.concatenate(audio_stft, axis=-1)
    elif model_version == 2: # W/ DC
        audio_stft = []
        for micIdx in range(audio_ch):
            mic_audio_stft = librosa.stft(audios[:, micIdx],
                                             n_fft=_stft_nfft,
                                             hop_length=_stft_hop_len,
                                             win_length=stft_win_len,
                                             # center=True
                                             center=True
                                             )
            audio_stft.append(mic_audio_stft[:, 1:])
        audio_stft = np.stack(audio_stft, axis=0)
        # for idxfr in range(frame_size):
        #     audio_ft_stft = []
        #     audios_fr = audios[hop_label_len * idxfr: hop_label_len * (idxfr + 1), :]
        #
        #     for micIdx in range(audio_ch):
        #         mic_audio_fr_stft = librosa.stft(audios_fr[:, micIdx],
        #                                          n_fft=_stft_nfft,
        #                                          hop_length=_stft_hop_len,
        #                                          win_length=stft_win_len,
        #                                          center=True
        #                                          )
        #         audio_ft_stft.append(mic_audio_fr_stft)
        #     audio_stft.append(np.stack(audio_ft_stft, axis=0))
        # audio_stft = np.concatenate(audio_stft, axis=-1)
    return audio_stft

def get_mel_spectrogram(linear_spectra, mel_wts, nb_mel_bins=64):
    num_ch = linear_spectra.shape[0]
    num_fr = linear_spectra.shape[-1]
    mel_feat = np.zeros((num_fr, nb_mel_bins, num_ch))
    for ch_cnt in range(num_ch):
        tmp_linear_spectra = np.transpose(linear_spectra[ch_cnt, :, :], (1, 0))
        mag_spectra = np.abs(tmp_linear_spectra)**2
        mel_spectra = np.dot(mag_spectra, mel_wts)
        log_mel_spectra = librosa.power_to_db(mel_spectra)
        mel_feat[:, :, ch_cnt] = log_mel_spectra
    # mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
    return np.transpose(mel_feat, (2, 1, 0))

def get_gcc(linear_spectra, len_cropped_cc=None, mic_pair_idx=None):
    num_ch = linear_spectra.shape[0]
    num_fr = linear_spectra.shape[-1]
    gcc_channels = nCr(num_ch, 2)
    if len_cropped_cc is None:
        len_cropped_cc = 960    # 20230828 IK, for DeepGCC
    gcc_feat = np.zeros((num_fr, len_cropped_cc, gcc_channels))

    if mic_pair_idx is None:
        mic_pair_idx = []
        for m in range(num_ch):
            for n in range(m + 1, num_ch):
                mic_pair_idx.append([m, n])

    for cnt, (m, n) in enumerate(mic_pair_idx):
        tmp_linear_spectra_1 = np.transpose(linear_spectra[m, :, :], (1, 0))
        tmp_linear_spectra_2 = np.transpose(linear_spectra[n, :, :], (1, 0))
        R = np.conj(tmp_linear_spectra_1) * tmp_linear_spectra_2
        cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
        cc = np.concatenate((cc[:, -len_cropped_cc//2:], cc[:, :len_cropped_cc//2]), axis=-1)
        gcc_feat[:, :, cnt] = cc
    return np.transpose(gcc_feat, (0, 2, 1))
    # cnt = 0
    # for m in range(num_ch):
    #     tmp_linear_spectra_1 = np.transpose(linear_spectra[m, :, :], (1, 0))
    #     for n in range(m+1, num_ch):
    #         tmp_linear_spectra_2 = np.transpose(linear_spectra[n, :, :], (1, 0))
    #         R = np.conj(tmp_linear_spectra_1) * tmp_linear_spectra_2
    #         cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
    #         cc = np.concatenate((cc[:, -len_cropped_cc//2:], cc[:, :len_cropped_cc//2]), axis=-1)
    #         gcc_feat[:, :, cnt] = cc
    #         cnt += 1
    # return np.transpose(gcc_feat, (2, 1, 0))