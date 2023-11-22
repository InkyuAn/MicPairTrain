import random
import sys
import os

import math
import numpy as np
import csv
import copy

import librosa
from torch.utils.data import Dataset
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import scipy.io.wavfile as wav

from apkit.apkit.basic import load_wav, load_metadata, stft, cola_hamming, cola_hamming, mel_freq_fbank_weight, empirical_cov_mat
from apkit.apkit.doa import angular_distance, azimuth_distance, vec2ae, load_pts_horizontal

from apkit.apkit.cc import gcc_phat_fbanks
from apkit.apkit.doa import azimuth_distance, load_pts_horizontal

from Dataset_Management.extract_labels import read_CSV, extract_diff_TDOA, extract_TDOA_labels, extract_DOA_labels
from utils.compute_stft import compute_stft, get_mel_spectrogram, get_gcc
import matplotlib.pyplot as plt
import get_param as parameter

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

# class dataset_batch_audios_for_seldnet(Dataset):
#     def __init__(self, audio_dir_list, gt_dir_list, params,                 
#                  out_type=0, sampling_data=False, out_dim=3
#                  ):
#         super(dataset_batch_audios_for_seldnet, self).__init__()        

#         self._out_dim = out_dim

#         self._fs = params['fs']
#         self._hop_len_s = params['hop_len_s']
#         self._hop_len = int(self._fs * self._hop_len_s)

#         self._label_hop_len_s = params['label_hop_len_s']
#         self._label_hop_len = int(self._fs * self._label_hop_len_s)
#         self._label_frame_res = self._fs / float(self._label_hop_len)
#         self._nb_label_frames_1s = int(self._label_frame_res)

#         self._win_len = 2 * self._hop_len
#         self._nfft = self._next_greater_power_of_2(self._win_len)
#         self._nb_mel_bins = params['nb_mel_bins']
#         self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T

#         self._eps = 1e-8
#         self._nb_channels = 4

#         # self._num_labels = len(params['unique_classes'])
#         self._saved_label_tdoa = params['saved_unique_classes_tdoa']
#         self._label_tdoa = params['unique_classes_tdoa']
#         self._label_doa = params['unique_classes_doa']
#         self._num_labels = len(params['unique_classes_doa'])  # IK 2022 01 01

#         self._audio_max_len_samples = int(params['batch_audio_len_s'] * self._fs)

#         self._max_feat_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len)))
#         self._max_label_frames = int(np.ceil(self._audio_max_len_samples / float(self._label_hop_len)))

#         # self._mic_positions = mic_positions
#         ''' Microphone array of DCASE dataset '''
#         mic_rad_dcase = 0.042  # 0.042 m = 4.2 cm
#         mic_positions_sph_dcase = [[45, 35], [-45, -35], [135, -35], [-135, 35]]
#         mic_pos_dcase = []
#         for mic_pos in mic_positions_sph_dcase:
#             azi_rad = mic_pos[0] * np.pi / 180
#             ele_rad = mic_pos[1] * np.pi / 180
#             tmp_label = np.cos(ele_rad)
#             x = np.cos(azi_rad) * tmp_label
#             y = np.sin(azi_rad) * tmp_label
#             z = np.sin(ele_rad)
#             mic_pos_dcase.append([x * mic_rad_dcase, y * mic_rad_dcase, z * mic_rad_dcase])

#         ''' Microphone array of DCASE dataset '''
#         mic_pos_sslr = [
#             [-0.0267, 0.0343, 0],
#             [-0.0267, -0.0343, 0],
#             [0.0313, 0.0343, 0],
#             [0.0313, -0.0343, 0]]

#         ''' Microphone array of Respeaker v2 dataset '''
#         mic_xy_respeaker = 0.02285  # 2.285 cm
#         mic_pos_respeaker = [
#             # [mic_xy_respeaker, -mic_xy_respeaker, 0],   # 4
#             # [mic_xy_respeaker, mic_xy_respeaker, 0],    # 3
#             # [-mic_xy_respeaker, mic_xy_respeaker, 0],   # 1
#             # [-mic_xy_respeaker, -mic_xy_respeaker, 0]   # 2
#             [-mic_xy_respeaker, mic_xy_respeaker, 0],  # 1
#             [-mic_xy_respeaker, -mic_xy_respeaker, 0],  # 2
#             [mic_xy_respeaker, mic_xy_respeaker, 0],  # 3
#             [mic_xy_respeaker, -mic_xy_respeaker, 0]  # 4
#         ]

#         self._mic_pair_idx = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

#         self._mic_pair_pos_dcase = []
#         self._mic_pair_pos_sslr = []
#         self._mic_pair_pos_respeaker = []
#         for p_idx in self._mic_pair_idx:
#             self._mic_pair_pos_dcase.append([mic_pos_dcase[p_idx[0]], mic_pos_dcase[p_idx[1]]])
#             self._mic_pair_pos_sslr.append([mic_pos_sslr[p_idx[0]], mic_pos_sslr[p_idx[1]]])
#             self._mic_pair_pos_respeaker.append([mic_pos_respeaker[p_idx[0]], mic_pos_respeaker[p_idx[1]]])

#         self._mic_center_pos_dcase = np.mean(np.asarray(mic_pos_dcase), axis=0)
#         self._mic_center_pos_sslr = np.mean(np.asarray(mic_pos_sslr), axis=0)
#         self._mic_center_pos_respeaker = np.mean(np.asarray(mic_pos_respeaker), axis=0)

#         self._out_type = out_type
#         '''
#             out_type == 0 : audio_out, gts_pair_out, gts_pair_xyz_out, data_dype 
#             out_type == 1 : audio_out, gts_pair_out, gts_pair_xyz_out, data_dype, audio_name

#             out_type == 11 : audio_out (STFT), gts_pair_out, gts_pair_xyz_out, data_dype (model_version: 1) 
#             out_type == 12 : audio_out (STFT), gts_pair_out, gts_pair_xyz_out, data_dype (model_version: 2)
#             out_type == 13 : audio_out (STFT), gts_pair_out, gts_pair_xyz_out, data_dype (model_version: 3)
#         '''
#         self._pts_horizontal = load_pts_horizontal()

#         tmp_audio_name_list = []
#         tmp_gt_name_list = []
#         for idx, audio_dir in enumerate(audio_dir_list):
#             tmp_audio_names = [audio_file_name for audio_file_name in os.listdir(audio_dir)]
#             audio_file_names = [os.path.join(audio_dir, audio_name) for audio_name in tmp_audio_names]
#             # self._audio_name_list.append(audio_file_names)
#             # self._audio_name_list += audio_file_names
#             tmp_audio_name_list += audio_file_names

#             gt_dir = gt_dir_list[idx]

#             gt_file_names = [os.path.join(gt_dir, audio_name.replace('audio', 'gtf').replace('.wav', '.csv'))
#                              for audio_name in tmp_audio_names]
#             # self._gt_name_list.append(gt_file_names)
#             # self._gt_name_list += gt_file_names
#             tmp_gt_name_list += gt_file_names

#         if sampling_data == False:
#             self._audio_name_list = tmp_audio_name_list
#             self._gt_name_list = tmp_gt_name_list

#             # print("Debugging")
#         else:
#             self._audio_name_list = []
#             self._gt_name_list = []

#             num_of_selected_files = 10
#             num_of_files = len(tmp_audio_name_list)
#             file_index_list = [idx for idx in range(num_of_files)]

#             # Shuffle
#             random.seed(10)
#             random.shuffle(file_index_list)

#             # Select 10 files
#             for idx in range(num_of_selected_files):
#                 idxFile = file_index_list[idx]

#                 self._audio_name_list.append(tmp_audio_name_list[idxFile])
#                 self._gt_name_list.append(tmp_gt_name_list[idxFile])

#             # print("Debugging")

#     def __getitem__(self, index):

#         # print("Test 1")
#         audio_file_name = self._audio_name_list[index]
#         gt_file_name = self._gt_name_list[index]

#         # print("Test 2")

#         data_type = 0
#         mic_pair_pos, mic_center_pos = None, None
#         if audio_file_name.find('DCASE') > 0:
#             mic_pair_pos = self._mic_pair_pos_dcase
#             mic_center_pos = self._mic_center_pos_dcase
#             data_type = 0
#         elif audio_file_name.find('sslr') > 0:
#             mic_pair_pos = self._mic_pair_pos_sslr
#             mic_center_pos = self._mic_center_pos_sslr
#             data_type = 1
#         # elif audio_file_name.find('Synthetic') > 0:
#         elif audio_file_name.find('Synt') > 0:
#             mic_pair_pos = self._mic_pair_pos_respeaker
#             mic_center_pos = self._mic_center_pos_respeaker
#             data_type = 2

#         ### Read audio file and compute the spectrogram
#         spect = self._get_spectrogram_for_file(audio_file_name)

#         # extract mel
#         mel_spect = self._get_mel_spectrogram(spect)

#         # extract gcc
#         gcc = self._get_gcc(spect)
#         feat = np.concatenate((mel_spect, gcc), axis=-1)
#         feat = np.transpose(feat, (2, 0, 1))

#         nFrames = self._nb_label_frames_1s
#         #######################################
#         ### Read CSV
#         # gts_info = []
#         numpy_gts = np.zeros((nFrames, self._out_dim * self._num_labels))
#         # encoded_like_gt_ovr_fr = np.zeros((nFrames, 360))
#         with open(gt_file_name, newline='') as csvfile:
#             reader = csv.reader(csvfile)
#             tmp_gts = list(reader)
#             # print("Test 4, len(tmp_gts): ", np.array(tmp_gts).shape)
#             for gt in tmp_gts:
#                 fix = int(gt[0])

#                 src_pos = np.array(list(map(float, gt[3:])), dtype=np.float)
#                 doa_vec = src_pos - mic_center_pos
#                 doa_vec = doa_vec / np.linalg.norm(doa_vec)

#                 ######################################################################################################
#                 # Modified, IK 2022 01 01
#                 class_idx = None
#                 for key in self._saved_label_tdoa:
#                     if gt[1] == self._saved_label_tdoa[key]:
#                         if key == 'femalescream' or key == 'malescream':
#                             class_idx = int(self._label_doa['scream'])
#                         elif key == 'femalespeech' or key == 'malespeech':
#                             class_idx = int(self._label_doa['speech'])
#                         else:
#                             class_idx = int(self._label_doa[key])

#                 # class_idx = int(gt[1])    # Prior version, IK 2022 01 01

#                 # numpy_gts[fix, class_idx * 3 : (class_idx+1) * 3] = doa_vec
#                 numpy_gts[fix, class_idx] = doa_vec[0]
#                 numpy_gts[fix, class_idx + self._num_labels] = doa_vec[1]
#                 if self._out_dim > 2:
#                     numpy_gts[fix, class_idx + self._num_labels*2] = doa_vec[2]

#         return feat.astype(np.float32), numpy_gts.astype(np.float32)
#         # astype(np.float32)
#     def __len__(self):
#         return len(self._audio_name_list)

#     @staticmethod
#     def _next_greater_power_of_2(x):
#         return 2 ** (x - 1).bit_length()

#     def _get_spectrogram_for_file(self, audio_path):
#         audio_in, fs = self._load_audio(audio_path)
       
#         audio_spec = self._spectrogram(audio_in)
#         return audio_spec

#     def _load_audio(self, audio_path):
#         fs, audio = wav.read(audio_path)
#         audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
#         if audio.shape[0] < self._audio_max_len_samples:
#             zero_pad = np.random.rand(self._audio_max_len_samples - audio.shape[0], audio.shape[1])*self._eps
#             audio = np.vstack((audio, zero_pad))
#         elif audio.shape[0] > self._audio_max_len_samples:
#             audio = audio[:self._audio_max_len_samples, :]
#         return audio, fs
#     def _spectrogram(self, audio_input):
#         _nb_ch = audio_input.shape[1]
#         nb_bins = self._nfft // 2
#         spectra = np.zeros((self._max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
#         for ch_cnt in range(_nb_ch):
#             stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
#                                         win_length=self._win_len, window='hann')
#             spectra[:, :, ch_cnt] = stft_ch[:, :self._max_feat_frames].T
#         return spectra
#     def _get_mel_spectrogram(self, linear_spectra):
#         mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
#         for ch_cnt in range(linear_spectra.shape[-1]):
#             mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
#             mel_spectra = np.dot(mag_spectra, self._mel_wts)
#             log_mel_spectra = librosa.power_to_db(mel_spectra)
#             mel_feat[:, :, ch_cnt] = log_mel_spectra
#         # mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
#         return mel_feat

#     def _get_gcc(self, linear_spectra):
#         gcc_channels = nCr(linear_spectra.shape[-1], 2)
#         gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
#         cnt = 0
#         for m in range(linear_spectra.shape[-1]):
#             for n in range(m+1, linear_spectra.shape[-1]):
#                 R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
#                 cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
#                 cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
#                 gcc_feat[:, :, cnt] = cc
#                 cnt += 1
#         return gcc_feat
#         # return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

### This is utilized in the DOA estimation stage
class dataset_batch_audios(Dataset):
    def __init__(self, audio_dir_list, gt_dir_list, params, #fs, hop_label_len_s,
                 # normalizer=None,                 
                 out_type=0, sampling_data=False,
                 stft_n_fft=960,
                 stft_hop_len=480,
                 stft_win_len=960,
                 out_dim=2,
                 stft_window=None
                 ):
        super(dataset_batch_audios, self).__init__()
        
        # self._fs = fs
        # self._hop_label_len_s = hop_label_len_s
        # fs = params['fs']
        fs = params['fs_Est_TDOA']
        self._fs = fs
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        self._hop_label_len_s = params['label_hop_len_s']
        self._hop_label_len = int(self._hop_label_len_s * self._fs)

        ### For TDOA labels having a higher resolution, 20220322
        increasing_weight = params['weight_fs_tdoa_label']   # 24000 Hz * 2 = 48000 Hz
        self._fs_tdoa_label = int(fs * increasing_weight)

        half_delay_len = params['half_delay_len']
        self._half_ndelays = int(half_delay_len * increasing_weight)     # prior: self._half_ndelays = half_delay_len
        self._ndelays = half_delay_len * 2 + 1

        self._pts_horizontal = load_pts_horizontal()
        self._sigma_sq = 0.005  # Sigma squared of Gaussian-like function

        ### Parameters of GCC-FB
        tmp_win_len = self._hop_label_len
        _FREQ_MAX = 8000
        _FREQ_MIN = 100
        self._zoom = 25
        self._nfbin = _FREQ_MAX * tmp_win_len // fs
        self._num_filter_bank = 40  # The number of filter banks of GCC-FB
        freq_bin_size = tmp_win_len
        self._freq = np.fft.fftfreq(freq_bin_size)[:self._nfbin]
        self._fbw = mel_freq_fbank_weight(self._num_filter_bank, self._freq, fs, fmax=_FREQ_MAX, fmin=_FREQ_MIN)

        ### Parameters for STFT
        self._fr_size = 10
        self._input_audio_len = self._fr_size * tmp_win_len  # 10 * 2400

        self._stft_nfft = stft_n_fft
        self._stft_hop_len = stft_hop_len
        self.stft_win_len = stft_win_len

        ### Parameters For mel-spectrum
        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self.stft_win_len)
        self._nb_mel_bins = params['nb_mel_bins']
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T
        self._max_feat_frames = int(np.ceil(self._input_audio_len / float(self._stft_hop_len)))

        ### Parameters For mel-spectrum (ours)
        self._mel_wts_ours = librosa.filters.mel(sr=self._fs, n_fft=self._stft_nfft, n_mels=self._nb_mel_bins).T

        ### Parameters for DCASE dataset
        self._out_dim = out_dim
        self._saved_label_tdoa = params['saved_unique_classes_tdoa']
        self._label_tdoa = params['unique_classes_tdoa']
        self._label_doa = params['unique_classes_doa']
        self._gt_dict = params['gt_dict']

        ''' Microphone array parameters, IK 20220517 '''
        self._mic_pos_tut_ca = params['mic_pos_tut_ca']

        self._mic_center_pos_dcase = params['mic_center_pos_dcase']
        self._mic_center_pos_sslr = params['mic_center_pos_sslr']
        self._mic_center_pos_tut_ca = params['mic_center_pos_tut_ca']
        self._mic_center_pos_respeaker = params['mic_center_pos_respeaker']
        self._mic_center_pos_8ch_cube = params['mic_center_pos_8ch_cube']
        self._mic_center_pos_8ch_circle = params['mic_center_pos_8ch_circle']

        self._mic_pair_idx = params['mic_pair_idx']

        self._mic_pair_pos_dcase = params['mic_pair_pos_dcase']
        self._mic_pair_pos_sslr = params['mic_pair_pos_sslr']
        self._mic_pair_pos_respeaker = params['mic_pair_pos_respeaker']
        self._mic_pair_pos_8ch_cube = params['mic_pair_pos_8ch_cube']
        self._mic_pair_pos_8ch_circle = params['mic_pair_pos_8ch_circle']


        self._mic_pos_respeaker = params['mic_pos_respeaker']

        self._out_type = out_type

        self._pts_horizontal = load_pts_horizontal()

        """
        0. alarm
        1. crying baby
        2. crash
        3. barking dog
        4. female scream
        5. female speech    --> Speech
        6. footsteps
        7. knocking on door
        8. male scream
        9. male speech      --> Speech
        10. ringing phone
        11. piano
        """
        # self._speech_class = ['5', '9']

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

    def __getitem__(self, index):

        audio_file_name = self._audio_name_list[index]
        gt_file_name = self._gt_name_list[index]

        ## Read Wav
        fs, audio_data = load_wav(audio_file_name)
        nFrames = audio_data.shape[1] // self._hop_label_len

        ## Set parameters corresponding to the dataset
        audio_ch, npairs, data_type = 0, 0, 0
        mic_pair_idx, mic_pair_pos, mic_center_pos = None, None, None
        if audio_file_name.find('dcase') > 0:
            mic_pair_idx = self._mic_pair_idx[0]
            mic_pair_pos = self._mic_pair_pos_dcase
            mic_center_pos = self._mic_center_pos_dcase
            audio_ch, npairs, data_type = 4, 6, 0
            
        elif audio_file_name.find('sslr') > 0:
            mic_pair_idx = self._mic_pair_idx[1]
            mic_pair_pos = self._mic_pair_pos_sslr
            mic_center_pos = self._mic_center_pos_sslr
            audio_ch, npairs, data_type = 4, 6, 1
            
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
            audio_ch, npairs, data_type = 4, 6, 2

            audio_data = audio_data[mic_indice, :]  # Sample 4-ch audios from 8-ch audios
                    

        ### Read CSV
        gts = read_CSV(gt_file_name)

        ### Generate labels of Pairs
        gts_diff = extract_diff_TDOA(gts, mic_center_pos, mic_pair_pos, self._half_ndelays,
                                     sampling_rate=self._fs_tdoa_label)
        gts_pair, gts_pair_xyz = extract_TDOA_labels(gts, gts_diff, nFrames, npairs, self._half_ndelays)

        ### Return Features and labels
        if self._out_type == -1:  ### Only return audio (Speech-oriented SSL)
            ### Encoding GTs of speech
            numpy_gts, encoded_like_gt_ovr_fr = extract_DOA_labels(gts, nFrames, mic_center_pos,
                                                                   dataset_flag=parameter.SSLR_DATASET)
            return audio_data.astype(np.float32), numpy_gts.astype(np.float32)
        elif self._out_type == -2:    ### Only return audio (SELD task)
            ### Encode GTs of all sound events
            numpy_gts, encoded_like_gt_ovr_fr = extract_DOA_labels(gts, nFrames, mic_center_pos, out_dim=self._out_dim,
                                                                   dataset_flag=parameter.DCASE2021_DATASET)

            return audio_data.astype(np.float32), numpy_gts.astype(np.float32)
        elif self._out_type < 10: ### Output for SSLR dataset
            ### Encoding GTs of speech
            numpy_gts, encoded_like_gt_ovr_fr = extract_DOA_labels(gts, nFrames, mic_center_pos, dataset_flag=parameter.SSLR_DATASET)

            ### Output: STFT (time-frequency), gt (x, y, z), gt-likelihood --- SSLR (MLP-GCC, GCC-PHAT-SM)
            if self._out_type == 0:
                win_len, hop_len = self._hop_label_len, self._hop_label_len
                tf = stft(audio_data, win_len, hop_len, window=cola_hamming, last_sample=True)
                tf = np.transpose(tf, [1, 0, 2])

                return tf.astype(np.complex64), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(np.float32)

            ### Output: GCC-FB, gt (x, y, z), gt-likelihood --- SSLR (CNN-GCCFB)
            elif self._out_type == 1:
                win_len, hop_len = self._hop_label_len, self._hop_label_len
                tf = stft(audio_data, win_len, hop_len, window=cola_hamming, last_sample=True)
                tf_forGCCFB = tf[:, :, :self._nfbin]
                ecov = empirical_cov_mat(tf_forGCCFB, fw=1, tw=1)
                fbcc = gcc_phat_fbanks(ecov, self._fbw, zoom=self._zoom, freq=self._freq)

                np_fbcc = np.zeros((len(fbcc), self._num_filter_bank, tf.shape[1], self._zoom * 2 + 1))
                for idx, idxMics in enumerate(fbcc):
                    np_fbcc[idx] = fbcc[idxMics]
                np_fbcc = np.transpose(np_fbcc, [2, 0, 3, 1])

                return np_fbcc.astype(np.float32), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(np.float32)

            ### Output: STFT pair, gt (x, y, z), gt_likelihood --> DCASE2021 (Ours)
            elif self._out_type == 2 or self._out_type == 3:

                stft_out_type = self._out_type - 1  # 2 --> 1 and 3 --> 2
                # Apply STFT
                # tmp_stft = self._compute_stft(self._out_type-10, np.transpose(audio_data, [1, 0]), 4)  # Check Audio_data shape
                tmp_stft = compute_stft(stft_out_type, np.transpose(audio_data, [1, 0]), audio_ch, self._fr_size, self._hop_label_len,
                     self._stft_nfft, self._stft_hop_len, self.stft_win_len)  # Check Audio_data shape

                audio_stft_pair = []
                # for idx, micIdx in enumerate(self._mic_pair_idx):
                for idx, micIdx in enumerate(mic_pair_idx):
                    tmp_audio_stft_pair = np.stack((tmp_stft[micIdx[0]], tmp_stft[micIdx[1]]), axis=0)
                    tmp_audio_stft_pair = np.concatenate((tmp_audio_stft_pair.real, tmp_audio_stft_pair.imag), axis=0)
                    audio_stft_pair.append(tmp_audio_stft_pair)
                audio_stft_pair = np.stack(audio_stft_pair, axis=0)

                stft_out = audio_stft_pair

                return stft_out.astype(np.float32), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(np.float32), gts_pair.astype(np.float32)

            ### Output: Mel spect + GCC-PHAT (Not used)
            elif self._out_type == 5:
                # Apply STFT
                tmp_stft = compute_stft(2, np.transpose(audio_data, [1, 0]), audio_ch, self._fr_size,
                                    self._hop_label_len,
                                    self._stft_nfft, self._stft_hop_len, self.stft_win_len)  # Check Audio_data shape

                audio_feat_pair = []
                for idx, micIdx in enumerate(mic_pair_idx):
                    tmp_audio_stft_pair = np.stack((tmp_stft[micIdx[0]], tmp_stft[micIdx[1]]), axis=0)

                    # Compute mel spectrogram
                    mel_spect = get_mel_spectrogram(tmp_audio_stft_pair, self._mel_wts_ours, nb_mel_bins=self._nb_mel_bins)

                    # extract gcc
                    gcc = get_gcc(tmp_audio_stft_pair, nb_mel_bins=self._nb_mel_bins)

                    feat = np.concatenate((mel_spect, gcc), axis=0)
                    audio_feat_pair.append(feat)
                audio_feat_pair = np.stack(audio_feat_pair, axis=0)

                stft_out = audio_feat_pair

                return stft_out.astype(np.float32), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(np.float32)
            ### Output: GCC (for DeepGCC), 20230830 IK
            elif self._out_type == 9:
                tmp_half_ndelays = 128 // 2

                # Apply STFT
                tmp_stft = compute_stft(2, np.transpose(audio_data, [1, 0]), audio_ch, self._fr_size,
                                        self._hop_label_len,
                                        self._stft_nfft, self._stft_hop_len,
                                        self.stft_win_len)  # Check Audio_data shape

                gcc = get_gcc(tmp_stft, len_cropped_cc=tmp_half_ndelays * 2, mic_pair_idx=mic_pair_idx)

                # return gcc.astype(np.float32), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(np.float32), gts_pair.astype(np.float32), data_type
                return gcc.astype(np.float32), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(
                    np.float32), gts_pair.astype(np.float32)

        else:   # Output for DCASE2021 dataset
            #######################################
            ### Encode GTs of all sound events
            numpy_gts, encoded_like_gt_ovr_fr = extract_DOA_labels(gts, nFrames, mic_center_pos, out_dim=self._out_dim,
                                                                   dataset_flag=parameter.DCASE2021_DATASET)

            ### Output: features, gt (x, y, z) --- DCASE2021 (SELD-net)
            if self._out_type == 10:    # DCASE
                spect = self._spectrogram(np.transpose(audio_data, (1, 0)))

                # extract mel
                mel_spect = self._get_mel_spectrogram(spect)

                # extract gcc
                gcc = self._get_gcc(spect)
                feat = np.concatenate((mel_spect, gcc), axis=-1)
                feat = np.transpose(feat, (2, 0, 1))

                return feat.astype(np.float32), numpy_gts.astype(np.float32)

            ### Output: STFT pair, gt (x, y, z) --- DCASE2021 (Ours, MLP layer)
            elif self._out_type == 11 or self._out_type == 12:
                stft_out_type = self._out_type - 10
                # Apply STFT
                # tmp_stft = self._compute_stft(self._out_type-10, np.transpose(audio_data, [1, 0]), 4)  # Check Audio_data shape
                tmp_stft = compute_stft(stft_out_type, np.transpose(audio_data, [1, 0]), audio_ch, self._fr_size,
                                        self._hop_label_len,
                                        self._stft_nfft, self._stft_hop_len,
                                        self.stft_win_len)  # Check Audio_data shape

                audio_stft_pair = []
                for idx, micIdx in enumerate(mic_pair_idx):
                    tmp_audio_stft_pair = np.stack((tmp_stft[micIdx[0]], tmp_stft[micIdx[1]]), axis=0)
                    tmp_audio_stft_pair = np.concatenate((tmp_audio_stft_pair.real, tmp_audio_stft_pair.imag), axis=0)
                    audio_stft_pair.append(tmp_audio_stft_pair)
                audio_stft_pair = np.stack(audio_stft_pair, axis=0)

                stft_out = audio_stft_pair

                return stft_out.astype(np.float32), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(np.float32), gts_pair.astype(np.float32)

            ### Output: Mel spect + GCC-PHAT (not used)
            elif self._out_type == 15:
                # Apply STFT
                tmp_stft = compute_stft(2, np.transpose(audio_data, [1, 0]), audio_ch, self._fr_size,
                                        self._hop_label_len,
                                        self._stft_nfft, self._stft_hop_len,
                                        self.stft_win_len)  # Check Audio_data shape

                audio_feat_pair = []
                for idx, micIdx in enumerate(mic_pair_idx):
                    tmp_audio_stft_pair = np.stack((tmp_stft[micIdx[0]], tmp_stft[micIdx[1]]), axis=0)

                    # Compute mel spectrogram
                    mel_spect = get_mel_spectrogram(tmp_audio_stft_pair, self._mel_wts_ours, nb_mel_bins=self._nb_mel_bins)

                    # extract gcc
                    gcc = get_gcc(tmp_audio_stft_pair, nb_mel_bins=self._nb_mel_bins)

                    feat = np.concatenate((mel_spect, gcc), axis=0)
                    audio_feat_pair.append(feat)
                audio_feat_pair = np.stack(audio_feat_pair, axis=0)

                stft_out = audio_feat_pair

                return stft_out.astype(np.float32), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(
                    np.float32)
            elif self._out_type == 19:
                tmp_half_ndelays = 128 // 2

                # Apply STFT
                tmp_stft = compute_stft(2, np.transpose(audio_data, [1, 0]), audio_ch, self._fr_size,
                                        self._hop_label_len,
                                        self._stft_nfft, self._stft_hop_len,
                                        self.stft_win_len)  # Check Audio_data shape

                gcc = get_gcc(tmp_stft, len_cropped_cc=tmp_half_ndelays * 2, mic_pair_idx=mic_pair_idx)

                return gcc.astype(np.float32), numpy_gts.astype(np.float32), encoded_like_gt_ovr_fr.astype(np.float32), gts_pair.astype(np.float32)

    def __len__(self):
        return len(self._audio_name_list)

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

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

        r_mic_pairs_pos = mic_pairs_pos[:, 1, :] - mic_pairs_pos[:, 0, :]   # (M, 3) array

        if doa.ndim == 1:
            diff = -np.einsum('ij,j->i', r_mic_pairs_pos, doa) / c
        else:
            assert doa.ndim == 2
            diff = -np.einsum('ij,kj->ki', r_mic_pairs_pos, doa) / c

        return diff * self._fs


    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = np.zeros((self._max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[:, :self._max_feat_frames].T
        return spectra
    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        # mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat
