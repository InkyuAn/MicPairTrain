### This source code is utilized for training TDOA features

import random
import sys
import os
import copy

import math
import numpy as np
import csv

import librosa
from torch.utils.data import Dataset
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from apkit.apkit.basic import load_wav, load_metadata, stft, cola_hamming
from apkit.apkit.doa import angular_distance, azimuth_distance, vec2ae, load_pts_horizontal

from utils.compute_stft import compute_stft, get_mel_spectrogram, get_gcc
from Dataset_Management.extract_labels import read_CSV, extract_diff_TDOA, extract_TDOA_labels
import get_param as parameter

# from sklearn.preprocessing import StandardScaler

### TUT_CA dataset (8-ch) should be not used with SSLR and DCASE datasets having 4-ch audios
class dataset_batch_audios(Dataset):
    def __init__(self, audio_dir_list, gt_dir_list, params,                 
                 batch_size=1, out_type=0,
                 sampling_data=False,
                 stft_n_fft=960,
                 stft_hop_len=480,
                 stft_win_len=960
                 ):
        super(dataset_batch_audios, self).__init__()

        # fs = params['fs']
        fs = params['fs_Est_TDOA']
        self._fs = fs
        self._hop_label_len_s = params['label_hop_len_s']
        self._hop_label_len = int(self._hop_label_len_s * self._fs)

        self._hop_len = self._hop_label_len
        self._win_len = self._hop_label_len
        # self._nfft = self._next_greater_power_of_2(self._win_len)

        self._batch_size = batch_size
        self._fr_size = 10
        self._input_audio_len = self._fr_size * self._win_len  # 10 * 2400

        ### For TDOA labels having a higher resolution, 20220322
        increasing_weight = params['weight_fs_tdoa_label']   # 24000 Hz * 2 = 48000 Hz
        self._fs_tdoa_label = int(fs * increasing_weight)

        # print("Debugging_20220323, fs_tdoa_label: ", self._fs_tdoa_label)    # For debugging, 20220323

        half_delay_len = params['half_delay_len']
        self._half_ndelays = int(half_delay_len * increasing_weight)     # prior: self._half_ndelays = half_delay_len
        self._ndelays = half_delay_len * 2 + 1

        # For STFT
        self._stft_nfft = stft_n_fft
        self._stft_hop_len = stft_hop_len
        self.stft_win_len = stft_win_len

        # For mel spectrogram
        self._nb_mel_bins = params['nb_mel_bins']
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._stft_nfft, n_mels=self._nb_mel_bins).T

        # For applying Gaussian Filter to TDoA label, 20230829 IK
        gauss_filter_size = 15
        gauss_filter_sigma = 1.25
        self._gauss_filter = torch.nn.Conv1d(1, 1, gauss_filter_size, stride=1, padding=gauss_filter_size // 2,
                                             bias=False).requires_grad_(False)

        def gauss(x, x0, sigma, H=0, A=1):
            return H + A * torch.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        tmp_gauss_filter = torch.arange(0, gauss_filter_size)
        tmp_gauss_filter = gauss(tmp_gauss_filter, gauss_filter_size // 2, gauss_filter_sigma)

        self._gauss_filter.weight.data[0, 0, :] = tmp_gauss_filter

        ''' Microphone array parameters, IK 20220517 '''
        self._mic_pos_tut_ca = params['mic_pos_tut_ca']

        self._mic_center_pos_dcase = params['mic_center_pos_dcase']
        self._mic_center_pos_sslr = params['mic_center_pos_sslr']
        self._mic_center_pos_tut_ca = params['mic_center_pos_tut_ca']
        self._mic_center_pos_respeaker = params['mic_center_pos_respeaker']

        self._mic_pair_idx = params['mic_pair_idx']

        self._mic_pair_pos_dcase = params['mic_pair_pos_dcase']
        self._mic_pair_pos_sslr = params['mic_pair_pos_sslr']
        self._mic_pair_pos_respeaker = params['mic_pair_pos_respeaker']

        ###
        self._out_type = out_type
        '''
            out_type == 0 : audio_out, gts_pair_out, gts_pair_xyz_out, data_dype 
            out_type == 1 : audio_out, gts_pair_out, gts_pair_xyz_out, data_dype, audio_name

            out_type == 11 : audio_out (STFT), gts_pair_out, gts_pair_xyz_out, data_dype (model_version: 1) 
            out_type == 12 : audio_out (STFT), gts_pair_out, gts_pair_xyz_out, data_dype (model_version: 2)
            out_type == 13 : audio_out (STFT), gts_pair_out, gts_pair_xyz_out, data_dype (model_version: 3)
        '''
        self._pts_horizontal = load_pts_horizontal()

        """
        < The 18 target sound classes >

        0: alarm --> 0
        1: crying baby --> 1
        2: crash --> 2
        3: barking dog --> 3
        4: female scream --> 4
        5: female speech (TUT) --> 5
        6: footsteps --> 6
        7. knocking on door (TUT) --> 7
        8. male scream --> 4
        9. male speech (TUT) --> 5
        10. ringing phone (TUT) --> 8
        11. piano --> 9

        12. clearthroat (clearing throat) (TUT)  --> 10
        13. cough (TUT) --> 11
        14. doorslam (slamming door) (TUT) --> 12
        15. drawer (TUT) --> 13
        16. keyboard (keyboard clicks) (TUT) --> 14
        17. keys Drop (keys dropped on desk) (TUT)  --> 15   
        18. Human laughter (TUT) --> 16
        19. page turn (paper page turning) (TUT) --> 17
        """

        self._labels = params['saved_unique_classes_tdoa']
        self._gt_dict = params['gt_dict']

        tmp_audio_name_list = []
        tmp_gt_name_list = []
        for idx, audio_dir in enumerate(audio_dir_list):
            tmp_audio_names = [audio_file_name for audio_file_name in os.listdir(audio_dir)]
            audio_file_names = [os.path.join(audio_dir, audio_name) for audio_name in tmp_audio_names]
            tmp_audio_name_list += audio_file_names

            gt_dir = gt_dir_list[idx]

            gt_file_names = [os.path.join(gt_dir, audio_name.replace('audio', 'gtf').replace('.wav', '.csv'))
                             for audio_name in tmp_audio_names]
            tmp_gt_name_list += gt_file_names

        if sampling_data == False:
            self._audio_name_list = tmp_audio_name_list
            self._gt_name_list = tmp_gt_name_list

        else:
            self._audio_name_list = []
            self._gt_name_list = []

            num_of_selected_files = 10
            num_of_files = len(tmp_audio_name_list)
            file_index_list = [idx for idx in range(num_of_files)]

            # Shuffle
            random.seed(10)
            random.shuffle(file_index_list)

            # Select 10 files
            for idx in range(num_of_selected_files):
                idxFile = file_index_list[idx]

                self._audio_name_list.append(tmp_audio_name_list[idxFile])
                self._gt_name_list.append(tmp_gt_name_list[idxFile])

            # print("Debugging")

    def __getitem__(self, index):

        audio_file_name = self._audio_name_list[index]
        gt_file_name = self._gt_name_list[index]

        # print("AF: ", audio_file_name)
        ## Read Wav
        fs, audio_data = load_wav(audio_file_name, ch_first=False)
        nsamples = audio_data.shape[0]
        nframe = nsamples // self._hop_label_len

        ## Set parameters corresponding to the dataset
        audio_ch, npairs, data_dype = 0, 0, 0
        mic_pair_idx, mic_pair_pos, mic_center_pos = None, None, None
        if audio_file_name.find('dcase') > 0:
            mic_pair_idx = self._mic_pair_idx[0]
            mic_pair_pos = self._mic_pair_pos_dcase
            mic_center_pos = self._mic_center_pos_dcase
            audio_ch, npairs, data_dype = 4, 6, 0
            
        elif audio_file_name.find('sslr') > 0:
            mic_pair_idx = self._mic_pair_idx[1]
            mic_pair_pos = self._mic_pair_pos_sslr
            mic_center_pos = self._mic_center_pos_sslr
            audio_ch, npairs, data_dype = 4, 6, 1
            
        elif audio_file_name.find('tut') > 0:
            ### Randomly select 4 microphones
            mic_indice = random.sample(range(8), 4)
            mic_indice = sorted(mic_indice)
            mic_pair_idx = self._mic_pair_idx[2]

            sampled_mic_pos_tut_ca = [self._mic_pos_tut_ca[mix] for mix in mic_indice]
            mic_pair_pos = []
            for idx, p_idx in enumerate(mic_pair_idx):
                mic_pair_pos.append([sampled_mic_pos_tut_ca[p_idx[0]], sampled_mic_pos_tut_ca[p_idx[1]]])

            mic_center_pos = self._mic_center_pos_tut_ca
            audio_ch, npairs, data_dype = 4, 6, 2

            audio_data = audio_data[:, mic_indice]  # Sample 4-ch audios from 8-ch audios
                    

        ### Read CSV
        gts = read_CSV(gt_file_name)

        if self._out_type > 10:

            # if self._out_type == 15:    # Return Mel spectrogram + GCC-PHAT
            #     # Apply STFT
            #     stft = compute_stft(2, audio_data, audio_ch, self._fr_size,
            #                         self._hop_label_len,
            #                         self._stft_nfft, self._stft_hop_len, self.stft_win_len)  # Check Audio_data shape
            #
            #     audio_feat_pair = []
            #     for idx, micIdx in enumerate(mic_pair_idx):
            #         tmp_audio_stft_pair = np.stack((stft[micIdx[0]], stft[micIdx[1]]), axis=0)
            #
            #         # Compute mel spectrogram
            #         mel_spect = get_mel_spectrogram(tmp_audio_stft_pair, self._mel_wts, nb_mel_bins=self._nb_mel_bins)
            #
            #         # extract gcc
            #         gcc = get_gcc(tmp_audio_stft_pair, nb_mel_bins=self._nb_mel_bins)
            #
            #         feat = np.concatenate((mel_spect, gcc), axis=0)
            #         audio_feat_pair.append(feat)
            #     audio_feat_pair = np.stack(audio_feat_pair, axis=0)
            #
            #     stft_out = audio_feat_pair
            #     gts_pair_out = gts_pair
            #     gts_pair_xyz_out = gts_pair_xyz
            #
            #     return stft_out, gts_pair_out, gts_pair_xyz_out, data_dype
            if self._out_type == 19:    # Return GCC-PHAT
                tmp_half_ndelays = 128 // 2

                gts_diff = extract_diff_TDOA(gts, mic_center_pos, mic_pair_pos, tmp_half_ndelays,
                                             sampling_rate=self._fs_tdoa_label)
                ### Make GT lable of mic pair
                gts_pair, gts_pair_xyz = extract_TDOA_labels(gts, gts_diff, nframe, npairs, tmp_half_ndelays)
                tmp_gts_pair = gts_pair[:, :, :, :-1]

                ### Apply Gaussian filter, 20230829
                num_label, fr_len, num_pair, _ = tmp_gts_pair.shape
                tmp_gts_pair = tmp_gts_pair.reshape((-1, 1, tmp_half_ndelays*2))
                tmp_gts_pair = self._gauss_filter(torch.from_numpy(tmp_gts_pair).to(dtype=torch.float)).to(dtype=torch.float32).numpy()
                tmp_gts_pair = tmp_gts_pair.reshape((num_label, fr_len, num_pair, tmp_half_ndelays*2))
                tmp_gts_pair = np.max(tmp_gts_pair, axis=0) # Max pooling on labels

                # Apply STFT
                stft = compute_stft(2, audio_data, audio_ch, self._fr_size,
                                    self._hop_label_len,
                                    self._stft_nfft, self._stft_hop_len, self.stft_win_len)  # Check Audio_data shape

                gcc = get_gcc(stft, len_cropped_cc=tmp_half_ndelays * 2, mic_pair_idx=mic_pair_idx)

                ### Stack Ground truth of mic pairs to fit the size with "gcc"
                tmp_gts_pair_out = np.zeros(gcc.shape)
                fr_offset = 5
                for fid in range(fr_len):
                    tmp_gts_pair_out[fid*fr_offset:fid*fr_offset+fr_offset, :, :] = np.repeat(np.expand_dims(tmp_gts_pair[fid, :, :], axis=0), fr_offset, axis=0)

                stft_out = gcc
                gts_pair_out = tmp_gts_pair_out
                gts_pair_xyz_out = gts_pair_xyz

                return stft_out, gts_pair_out, gts_pair_xyz_out, data_dype
            else:   # Return STFT spectra
                gts_diff = extract_diff_TDOA(gts, mic_center_pos, mic_pair_pos, self._half_ndelays,
                                             sampling_rate=self._fs_tdoa_label)
                ### Make GT lable of mic pair
                gts_pair, gts_pair_xyz = extract_TDOA_labels(gts, gts_diff, nframe, npairs, self._half_ndelays)

                # Apply STFT
                stft = compute_stft(self._out_type - 10, audio_data, audio_ch, self._fr_size,
                                    self._hop_label_len,
                                    self._stft_nfft, self._stft_hop_len, self.stft_win_len)  # Check Audio_data shape

                audio_stft_pair = []
                for idx, micIdx in enumerate(mic_pair_idx):
                    tmp_audio_stft_pair = np.stack((stft[micIdx[0]], stft[micIdx[1]]), axis=0)
                    tmp_audio_stft_pair = np.concatenate((tmp_audio_stft_pair.real, tmp_audio_stft_pair.imag), axis=0)
                    # tmp_audio_stft_pair = np.concatenate((np.absolute(tmp_audio_stft_pair), np.angle(tmp_audio_stft_pair)), axis=0)
                    audio_stft_pair.append(tmp_audio_stft_pair)
                audio_stft_pair = np.stack(audio_stft_pair, axis=0)

                stft_out = audio_stft_pair
                gts_pair_out = gts_pair
                gts_pair_xyz_out = gts_pair_xyz

                # return audio_stft_pair, gts_pair_out, gts_pair_xyz_out, data_dype
                return stft_out, gts_pair_out, gts_pair_xyz_out, data_dype

    def __len__(self):
        return len(self._audio_name_list)

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    # def _spectrogram(self, audio_input, _nb_ch, _nFrame):
    #     nb_bins = self._nfft // 2
    #     spectra = np.zeros((_nFrame, nb_bins + 1, _nb_ch), dtype=complex)
    #     for ch_cnt in range(_nb_ch):
    #         stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
    #                                     win_length=self._win_len, window='hann')
    #         spectra[:, :, ch_cnt] = stft_ch[:, :_nFrame].T
    #     return spectra

    def _compute_delay(self, mic_pairs_pos, doa, c=340):
        """

        :param mic_pairs_pos: microphone pair's positions, (M, 2, 3) array,
                              M is the unique number of microphone pair
        :param doa: Normalized direction of arrival, (3,) array or (N, 3) array
                    N is the number of sources
        :param c: speed of sound (m/s)

        :return:
        """
        mic_pairs_pos = np.array(mic_pairs_pos)
        doa = np.array(doa)

        r_mic_pairs_pos = mic_pairs_pos[:, 1, :] - mic_pairs_pos[:, 0, :]  # (M, 3) array

        if doa.ndim == 1:
            diff = -np.einsum('ij,j->i', r_mic_pairs_pos, doa) / c
        else:
            assert doa.ndim == 2
            diff = -np.einsum('ij,kj->ki', r_mic_pairs_pos, doa) / c

        return diff * self._fs

    # def _compute_stft(self, model_version, audios, audio_ch):
    #
    #     frame_size = self._fr_size
    #     hop_label_len = self._hop_label_len
    #     audio_stft = []
    #     # if model_version == 1:
    #     #     for micIdx in range(audio_ch):
    #     #         mic_audio = np.copy(audios[:, micIdx])
    #     #         mic_audio_stft = librosa.stft(mic_audio,
    #     #                                       n_fft=self._stft_nfft,
    #     #                                       hop_length=self._stft_hop_len,
    #     #                                       win_length=self.stft_win_len,
    #     #                                       center=True
    #     #                                       )
    #     #         audio_stft.append(mic_audio_stft[1:, 1:])  # W/o DC and W/o First Frame
    #     #     audio_stft = np.stack(audio_stft, axis=0)
    #     # elif model_version == 2:
    #     if model_version == 1:      # W/O DC
    #         for idxfr in range(frame_size):
    #             audio_ft_stft = []
    #             audios_fr = audios[hop_label_len * idxfr: hop_label_len * (idxfr + 1), :]
    #
    #             for micIdx in range(audio_ch):
    #                 mic_audio_fr_stft = librosa.stft(audios_fr[:, micIdx],
    #                                               n_fft=self._stft_nfft,
    #                                               hop_length=self._stft_hop_len,
    #                                               win_length=self.stft_win_len,
    #                                               center=True
    #                                               )
    #                 audio_ft_stft.append(mic_audio_fr_stft[1:, :])  # W/o DC
    #             audio_stft.append(np.stack(audio_ft_stft, axis=0))
    #         audio_stft = np.concatenate(audio_stft, axis=-1)
    #     elif model_version == 2:    # W/ DC
    #         for idxfr in range(frame_size):
    #             audio_ft_stft = []
    #             audios_fr = audios[hop_label_len * idxfr: hop_label_len * (idxfr + 1), :]
    #
    #             for micIdx in range(audio_ch):
    #                 mic_audio_fr_stft = librosa.stft(audios_fr[:, micIdx],
    #                                               n_fft=self._stft_nfft,
    #                                               hop_length=self._stft_hop_len,
    #                                               win_length=self.stft_win_len,
    #                                               center=True
    #                                               )
    #                 audio_ft_stft.append(mic_audio_fr_stft)  # W/ DC
    #             audio_stft.append(np.stack(audio_ft_stft, axis=0))
    #         audio_stft = np.concatenate(audio_stft, axis=-1)
    #     # elif model_version == 3:
    #     #     for micIdx in range(audio_ch):
    #     #         mic_audio = np.copy(audios[:, micIdx])
    #     #         mic_audio_stft = librosa.stft(mic_audio,
    #     #                                          n_fft=self._stft_nfft,
    #     #                                          hop_length=self._stft_hop_len,
    #     #                                          win_length=self.stft_win_len,
    #     #                                          center=False
    #     #                                          )
    #     #         audio_stft.append(mic_audio_stft)
    #     #     audio_stft = np.stack(audio_stft, axis=0)
    #     #     audio_stft = audio_stft[:, 1:, :]  # W/o DC
    #     return audio_stft

    # def _get_spectrogram_for_file(self, audio_in):
    #     audio_spec = self._spectrogram(audio_in)
    #     return audio_spec
    #
    # def _spectrogram(self, audio_input):
    #     _nb_ch = audio_input.shape[1]
    #     nb_bins = self._nfft // 2
    #     spectra = np.zeros((self._max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
    #     for ch_cnt in range(_nb_ch):
    #         stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft,
    #                                     hop_length=self._stft_hop_len,
    #                                     win_length=self.stft_win_len, window='hann')
    #         spectra[:, :, ch_cnt] = stft_ch[:, :self._max_feat_frames].T
    #     return spectra
    #
    # def _get_mel_spectrogram(self, linear_spectra):
    #     mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
    #     for ch_cnt in range(linear_spectra.shape[-1]):
    #         mag_spectra = np.abs(linear_spectra[:, :, ch_cnt]) ** 2
    #         mel_spectra = np.dot(mag_spectra, self._mel_wts)
    #         log_mel_spectra = librosa.power_to_db(mel_spectra)
    #         mel_feat[:, :, ch_cnt] = log_mel_spectra
    #     # mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
    #     return mel_feat
    #
    # def _get_phase_mel_spectrogram(self, pair_diff_spectra):
    #     mel_phase = np.dot(pair_diff_spectra, self._mel_wts)
    #     return mel_phase