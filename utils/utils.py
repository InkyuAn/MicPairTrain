import os
import sys

import numpy as np
from torch.utils.data import DataLoader

import argparse

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from Dataset_Management.extract_labels import compute_delay
import get_param as parameter

# For the first stage (TDOA estimation)
import Dataset_Management.dataloader_batch_audios_pairs_multi_labels

# For the second stage (DOA estimation)
# import Dataset_Management.dataloader_batch_audios

from utils.dataloaders import get_datasetloader

# Deep models
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Mic_Pair_Train.models.CRNN_stft import CRNNpair
from Mic_Pair_Train.models.Robust_TDoA_HiFTA import RobustTDoA_HiFTA
from Mic_Pair_Train.models.Robust_TDoA_Transformer import RobustTDoA_Trans

from Mic_Pair_Train.models.MLP_DOA import MLP_DOA
from Mic_Pair_Train.models.MLP_DOA_v2 import MLP_DOA_v2

from Mic_Pair_Train.models.DeepGCC import deep_gcc

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def compute_diff_TDOA(grid_vertex, mic_center_pos, mic_pair_pos, num_half_delays, sampling_rate=24000):
    gts_xyz = grid_vertex
    if len(gts_xyz) > 0:
        # normalization
        doa_vec = np.array(gts_xyz) - np.repeat(np.expand_dims(mic_center_pos, axis=0), repeats=len(gts_xyz),
                                                axis=0)
        norm_doa_vec = np.linalg.norm(doa_vec, axis=-1)
        doa_vec = doa_vec / np.repeat(np.expand_dims(norm_doa_vec, axis=1), repeats=3, axis=1)

        diff = compute_delay(mic_pair_pos, doa_vec, fs=sampling_rate)
        return np.around(diff).astype(dtype=np.int) + num_half_delays
    else:
        return np.empty([0, 0])

def compute_diff_TDOA_float(grid_vertex, mic_center_pos, mic_pair_pos, num_half_delays, sampling_rate=24000):
    gts_xyz = grid_vertex
    if len(gts_xyz) > 0:
        # normalization
        doa_vec = np.array(gts_xyz) - np.repeat(np.expand_dims(mic_center_pos, axis=0), repeats=len(gts_xyz),
                                                axis=0)
        norm_doa_vec = np.linalg.norm(doa_vec, axis=-1)
        doa_vec = doa_vec / np.repeat(np.expand_dims(norm_doa_vec, axis=1), repeats=3, axis=1)

        diff = compute_delay(mic_pair_pos, doa_vec, fs=sampling_rate)
        return diff + num_half_delays
    else:
        return np.empty([0, 0])

def set_audio_parameters(fs):
    if fs == 24000:     # 24 kHz
        fft_length = 960
        hop_length = 480  # 0.02 second
        win_length = 960  # 0.04 second
        audio_length_for_1s = 24000
    elif fs == 48000:   # 48 kHz
        fft_length = 1920
        hop_length = 960  # 0.02 second
        win_length = 1920  # 0.04 second
        audio_length_for_1s = 48000
    else:
        fft_length = 960
        hop_length = 480  # 0.02 second
        win_length = 960  # 0.04 second
        audio_length_for_1s = 24000

    return  fft_length, hop_length, win_length, audio_length_for_1s

### IK, 20221124, modify computing a STFT signal
def set_STFT_parameters(stage_flag, fft_length, hop_length, audio_length_for_1s, model_version, num_frame):
    stft_h_size, stft_w_size = 0, 0
    stft_out_version = 0
    if model_version == 1 or model_version == 2 or model_version == 5:
        # stft_w_size = (1 + (audio_length_for_1s // num_frame) // hop_length) * num_frame  # Time
        stft_w_size = audio_length_for_1s // hop_length # Temporal
        stft_h_size = fft_length // 2  # Frequency

        if stage_flag == 0: # First stage (TDOA training stage)
            stft_out_version = 11
        elif stage_flag == 1:   # Second stage (DOA training stage)
            stft_out_version = (11, 2)  # (DCASE2021 (speeh + others), SSLR (only speech))
    elif model_version == 3 or model_version == 4 \
            or model_version == 10\
            or model_version == 20:
        # stft_w_size = (1 + (audio_length_for_1s // num_frame) // hop_length) * num_frame  # Time
        stft_w_size = audio_length_for_1s // hop_length  # Temporal
        stft_h_size = fft_length // 2 + 1 # Frequency

        if stage_flag == 0:  # First stage (TDOA training stage)
            stft_out_version = 12
        elif stage_flag == 1:  # Second stage (DOA training stage)
            stft_out_version = (12, 3)  # (DCASE2021 (speeh + others), SSLR (only speech))

    elif model_version == 9:    ### Prior work (Deep-GCC)
        # stft_w_size = (1 + (audio_length_for_1s // num_frame) // hop_length) * num_frame  # Time
        stft_w_size = audio_length_for_1s // hop_length  # Temporal
        stft_h_size = fft_length // 2 + 1  # Frequency

        if stage_flag == 0:  # First stage (TDOA training stage)
            stft_out_version = 19
        elif stage_flag == 1:  # Second stage (DOA training stage)
            stft_out_version = (19, 9)  # (DCASE2021 (speeh + others), SSLR (only speech))

    return stft_w_size, stft_h_size, stft_out_version

def get_train_dataset_1st_stage(USE_SSLR, USE_DCASE, USE_TUT,
                      params,
                      stft_out_version, fft_length, hop_length, win_length, batch_size,
                      num_worker_dataloader=8, sampling_data=False):
    file_batch_size = 1

    audioDir_dcase2021_train, gtDir_dcase2021_train = parameter.get_dataset_dir_list(parameter.DCASE2021_DATASET,
                                                                                     parameter.IS_TRAIN)
    audioDir_sslr_train, gtDir_sslr_train = parameter.get_dataset_dir_list(parameter.SSLR_DATASET,
                                                                           parameter.IS_TRAIN)
    audioDir_tut_train, gtDir_tut_train = parameter.get_dataset_dir_list(parameter.TUT_CA_DATASET,
                                                                         parameter.IS_TRAIN)

    ''' Load DCASE and SSLR dataset (Training) '''
    train_dataset_audio, train_dataset_label = None, None
    if USE_SSLR and USE_DCASE and USE_TUT:
        train_dataset_audio = audioDir_dcase2021_train + audioDir_sslr_train + audioDir_tut_train
        train_dataset_label = gtDir_dcase2021_train + gtDir_sslr_train + gtDir_tut_train
    elif USE_SSLR and USE_DCASE:
        train_dataset_audio = audioDir_dcase2021_train + audioDir_sslr_train
        train_dataset_label = gtDir_dcase2021_train + gtDir_sslr_train
    elif USE_SSLR:
        train_dataset_audio = audioDir_sslr_train
        train_dataset_label = gtDir_sslr_train
    elif USE_DCASE:
        train_dataset_audio = audioDir_dcase2021_train
        train_dataset_label = gtDir_dcase2021_train

    train_dataset = Dataset_Management.dataloader_batch_audios_pairs_multi_labels.dataset_batch_audios(train_dataset_audio, train_dataset_label, params,
                                         # fs, hop_label_len_s, half_delay_len,
                                         batch_size=file_batch_size, sampling_data=sampling_data,
                                         out_type=stft_out_version,
                                         stft_n_fft=fft_length,
                                         stft_hop_len=hop_length,
                                         stft_win_len=win_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_worker_dataloader, pin_memory=True)

    return train_dataloader

def get_test_dataset_1st_stage(dataset_type,
                                 params,
                                 stft_out_version, fft_length, hop_length, win_length, batch_size,
                                 num_worker_dataloader=8,
                                 sampling_data=False
                                 ):
    file_batch_size = 1

    out_dataloader_test = None
    if dataset_type == 0:   # DCASE
        audioDir_dcase2021_test, gtDir_dcase2021_test = parameter.get_dataset_dir_list(parameter.DCASE2021_DATASET,
                                                                                       parameter.IS_TEST)
        ''' Load DCASE and SSLR dataset (Testing) '''
        DCASE2021_dataset_test = Dataset_Management.dataloader_batch_audios_pairs_multi_labels.dataset_batch_audios(audioDir_dcase2021_test, gtDir_dcase2021_test, params,
                                                      batch_size=file_batch_size, sampling_data=sampling_data,
                                                      out_type=stft_out_version,
                                                      stft_n_fft=fft_length,
                                                      stft_hop_len=hop_length,
                                                      stft_win_len=win_length)
        DCASE2021_dataloader_test = DataLoader(DCASE2021_dataset_test, batch_size=batch_size, shuffle=False,
                                               num_workers=num_worker_dataloader, pin_memory=True)
        out_dataloader_test = DCASE2021_dataloader_test
    elif dataset_type == 1: # SSLR
        audioDir_sslr_test, gtDir_sslr_test = parameter.get_dataset_dir_list(parameter.SSLR_DATASET,
                                                                             parameter.IS_TEST)

        SSLR_dataset_test = Dataset_Management.dataloader_batch_audios_pairs_multi_labels.dataset_batch_audios(audioDir_sslr_test, gtDir_sslr_test, params,
                                                 batch_size=file_batch_size, sampling_data=sampling_data,
                                                 out_type=stft_out_version,
                                                 stft_n_fft=fft_length,
                                                 stft_hop_len=hop_length,
                                                 stft_win_len=win_length)
        SSLR_dataloader_test = DataLoader(SSLR_dataset_test, batch_size=batch_size, shuffle=False,
                                          num_workers=num_worker_dataloader, pin_memory=True)
        out_dataloader_test = SSLR_dataloader_test
    elif dataset_type == 2: # TUT
        audioDir_tut_test, gtDir_tut_test = parameter.get_dataset_dir_list(parameter.TUT_CA_DATASET,
                                                                           parameter.IS_TEST)

        TUT_dataset_test = Dataset_Management.dataloader_batch_audios_pairs_multi_labels.dataset_batch_audios(audioDir_tut_test, gtDir_tut_test, params,
                                                batch_size=file_batch_size, sampling_data=sampling_data,
                                                out_type=stft_out_version,
                                                stft_n_fft=fft_length,
                                                stft_hop_len=hop_length,
                                                stft_win_len=win_length)
        TUT_dataloader_test = DataLoader(TUT_dataset_test, batch_size=batch_size, shuffle=False,
                                         num_workers=num_worker_dataloader, pin_memory=True)
        out_dataloader_test = TUT_dataloader_test

    return out_dataloader_test

def get_all_dataset_2nd_stage(dataset_type,
                                 params,
                              stft_out_version, fft_length, hop_length, win_length, batch_size,
                              out_dim = 3,
                              num_worker_dataloader = 8,
                              sampling_data = False
                              ):

    ### Bypassing
    train_dataloader, out_test_dataloader = get_datasetloader(dataset_type, params,
                                                              stft_out_version, fft_length, hop_length, win_length, batch_size,
                                                              out_dim, num_worker_dataloader, sampling_data)

    return train_dataloader, out_test_dataloader
    # out_test_dataloader = None
    # train_dataloader = None
    # if dataset_type == 0:   # DCASE
    #     audioDir_dcase2021_train, gtDir_dcase2021_train = parameter.get_dataset_dir_list(parameter.DCASE2021_DATASET,
    #                                                                                      parameter.IS_TRAIN)
    #     audioDir_dcase2021_test, gtDir_dcase2021_test = parameter.get_dataset_dir_list(parameter.DCASE2021_DATASET,
    #                                                                                    parameter.IS_TEST)
    #
    #     train_dataset_audio = audioDir_dcase2021_train
    #     train_dataset_label = gtDir_dcase2021_train
    #
    #     train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio, train_dataset_label, params,
    #                                          out_type=stft_out_version, sampling_data=sampling_data, out_dim=out_dim,
    #                                          stft_n_fft=fft_length,
    #                                          stft_hop_len=hop_length,
    #                                          stft_win_len=win_length)
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                   num_workers=num_worker_dataloader, pin_memory=True)
    #
    #     ''' Load SSLR dataset (Testing) '''
    #     DCASE2021_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(audioDir_dcase2021_test, gtDir_dcase2021_test, params,
    #                                                   out_type=stft_out_version, sampling_data=sampling_data,
    #                                                   out_dim=out_dim,
    #                                                   stft_n_fft=fft_length,
    #                                                   stft_hop_len=hop_length,
    #                                                   stft_win_len=win_length)
    #     DCASE2021_dataloader_test = DataLoader(DCASE2021_dataset_test, batch_size=batch_size, shuffle=False,
    #                                            num_workers=num_worker_dataloader, pin_memory=True)
    #     out_test_dataloader = DCASE2021_dataloader_test
    #
    # elif dataset_type == 1:   # SSLR
    #     audioDir_sslr_train, gtDir_sslr_train = parameter.get_dataset_dir_list(parameter.SSLR_DATASET,
    #                                                                            parameter.IS_TRAIN)
    #     audioDir_sslr_test, gtDir_sslr_test = parameter.get_dataset_dir_list(parameter.SSLR_DATASET,
    #                                                                          parameter.IS_TEST)
    #
    #     train_dataset_audio = audioDir_sslr_train
    #     train_dataset_label = gtDir_sslr_train
    #
    #     train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio, train_dataset_label, params,
    #                                          out_type=stft_out_version, sampling_data=sampling_data,
    #                                          stft_n_fft=fft_length,
    #                                          stft_hop_len=hop_length,
    #                                          stft_win_len=win_length)
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                   num_workers=num_worker_dataloader, pin_memory=True)
    #
    #     ''' Load SSLR dataset (Testing) '''
    #     SSLR_dataset_test = Dataset_Management.dataloader_batch_audios .dataset_batch_audios(audioDir_sslr_test, gtDir_sslr_test, params,
    #                                              out_type=stft_out_version, sampling_data=sampling_data,
    #                                              stft_n_fft=fft_length,
    #                                              stft_hop_len=hop_length,
    #                                              stft_win_len=win_length)
    #     SSLR_dataloader_test = DataLoader(SSLR_dataset_test, batch_size=batch_size, shuffle=False,
    #                                       num_workers=num_worker_dataloader, pin_memory=True)
    #     out_test_dataloader = SSLR_dataloader_test
    # elif dataset_type == 2:  # TUT
    #     print("Not used in the second stage ... ")
    #
    # elif dataset_type == 3: # Synthetic dataset (dry signals: DCASE 2021)
    #     ''' For Synthetic dataset '''
    #     audioDir_synt2102_train, gtDir_synt2102_train = parameter.get_dataset_dir_list(parameter.SYNTHETIC_2102_DATASET,
    #                                                                                    parameter.IS_TRAIN)
    #
    #     audioDir_synt2102_test, gtDir_synt2102_test = parameter.get_dataset_dir_list(parameter.SYNTHETIC_2102_DATASET,
    #                                                                                  parameter.IS_TEST)
    #
    #     ''' Load Syntetic dataset (Training) '''
    #
    #     train_dataset_audio = audioDir_synt2102_train
    #     train_dataset_label = gtDir_synt2102_train
    #
    #     train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio,
    #                                                                                     train_dataset_label, params,
    #                                                                                     out_type=stft_out_version,
    #                                                                                     sampling_data=sampling_data,
    #                                                                                     out_dim=out_dim,
    #                                                                                     stft_n_fft=fft_length,
    #                                                                                     stft_hop_len=hop_length,
    #                                                                                     stft_win_len=win_length)
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                   num_workers=num_worker_dataloader, pin_memory=True)
    #
    #     ''' Load Synthetic dataset (Testing) '''
    #     Synt2102_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(audioDir_synt2102_test, gtDir_synt2102_test, params,
    #                                                  out_type=stft_out_version, sampling_data=sampling_data,
    #                                                  out_dim=out_dim,
    #                                                  stft_n_fft=fft_length,
    #                                                  stft_hop_len=hop_length,
    #                                                  stft_win_len=win_length)
    #     Synt2102_dataloader_test = DataLoader(Synt2102_dataset_test, batch_size=batch_size, shuffle=False,
    #                                           num_workers=num_worker_dataloader, pin_memory=True)
    #     out_test_dataloader = Synt2102_dataloader_test
    #
    #
    # return train_dataloader, out_test_dataloader

def select_model_1st_stage(model_version,
                           model_parameters):
    model = None
    model_name = None
    if model_version == 1:  # CRNN
        print(" - CRNN model is selected ...")
        model = CRNNpair(out_size=model_parameters['delay_len'],
                         cnn_filter_num=model_parameters['mlp_hidden_dim'], rnn_size=model_parameters['mlp_hidden_dim'], fc_size=model_parameters['mlp_hidden_dim'],
                         cnn_t_pool_size=(2, 2, 1),
                         cnn_f_pool_size=(4, 4, 2),
                         # pads=((1, 1), (1, 1), (1, 1)),
                         pads=((1, 0), (1, 1), (1, 0)),
                         num_labels=model_parameters['num_TDOA_labels'],
                         return_features=model_parameters['return_features'],
                         use_middle_fc=True,
                         fs=model_parameters['fs']
                         )
        model_name = 'CRNN'
    if model_version == 5:  # CRNN
        print(" - CRNN model is selected ...")
        model = CRNNpair(out_size=model_parameters['delay_len'],
                         cnn_filter_num=model_parameters['mlp_hidden_dim'], rnn_size=256, fc_size=model_parameters['mlp_hidden_dim'],
                         cnn_t_pool_size=(2, 2, 1),
                         cnn_f_pool_size=(8, 8, 2),
                         # pads=((1, 1), (1, 1), (1, 1)),
                         pads=((1, 0), (1, 1), (0, 0)),
                         num_labels=model_parameters['num_TDOA_labels'],
                         return_features=model_parameters['return_features'],
                         use_middle_fc=False,
                         fs=model_parameters['fs']
                         )
        model_name = 'CRNN_v2'
    elif model_version == 2:
        print(" - rTDOA w/ HiFTA w/o LMFB depth %d model is selected ..." % model_parameters['hifta_depth'])
        model = RobustTDoA_HiFTA(
            stft_size=(model_parameters['stft_h_size'], model_parameters['stft_w_size']),
            stft_ch=2 * 2,  # Audio ch. * (Mag. & Phase spectrums)
            patch_w_num=model_parameters['num_frame'],
            pixel_h_num=model_parameters['pixel_h_num'],
            ###
            patch_dim=model_parameters['patch_dim'],
            pixel_dim=model_parameters['pixel_dim'],
            depth=model_parameters['hifta_depth'],
            heads=model_parameters['hifta_heads'],
            dim_head=model_parameters['hifta_dim_head'],
            delay_len=model_parameters['delay_len'],
            mlp_head_dim=model_parameters['mlp_hidden_dim'],
            ff_dropout=0.1,
            attn_dropout=0.1,
            w_hifta=model_parameters['w_hifta'],
            num_labels=model_parameters['num_TDOA_labels'],
            return_features=model_parameters['return_features'],
            use_mel_filter=False,
            fs=model_parameters['fs']
        )
        model_name = 'HiFTA_depth%d' % model_parameters['hifta_depth']
    elif model_version == 3:  # CRNN
        ### TODO 20220602 IK, Need to be modified ... CRNN model may contain errors, The CRNN model architecture needs to be modified.
        model = CRNNpair(out_size=model_parameters['delay_len'],
                         cnn_filter_num=model_parameters['mlp_hidden_dim'], rnn_size=model_parameters['mlp_hidden_dim'], fc_size=model_parameters['mlp_hidden_dim'],
                         cnn_t_pool_size=(2, 2, 1),
                         cnn_f_pool_size=(4, 4, 4),
                         pads=((1, 0), (1, 1), (1, 0)),
                         num_labels=model_parameters['num_TDOA_labels'],
                         stft_h_size=model_parameters['stft_h_size'],
                         return_features=model_parameters['return_features'],
                         use_mel_filter=True,
                         use_middle_fc=False,
                         fs=model_parameters['fs']
                         )
        model_name = 'CRNN_LMFB'
        print(" - CRNN_LMFB model is selected ...")

    # elif model_version == 4:
    #     print(" - TNT_lmf model is selected ...")
    #     model = TNT_stft(
    #         stft_size=(model_parameters['stft_h_size'], model_parameters['stft_w_size']),
    #         stft_ch=2 * 2,  # Audio ch. * (Mag. & Phase spectrums)
    #         patch_w_num=model_parameters['num_frame'],
    #         pixel_h_num=model_parameters['pixel_h_num'],
    #         ###
    #         patch_dim=model_parameters['patch_dim'],
    #         pixel_dim=model_parameters['pixel_dim'],
    #         depth=model_parameters['tnt_depth'],
    #         heads=model_parameters['tnt_heads'],
    #         dim_head=model_parameters['tnt_dim_head'],
    #         delay_len=model_parameters['delay_len'],
    #         mlp_head_dim=model_parameters['mlp_hidden_dim'],
    #         ff_dropout=0.1,
    #         attn_dropout=0.1,
    #         w_hifta=model_parameters['w_hifta'],
    #         num_labels=model_parameters['num_TDOA_labels'],
    #         return_features=model_parameters['return_features'],
    #         use_mel_filter=True,
    #         fs=model_parameters['fs']
    #     )
    #     model_name = 'TNT_lmf'

    # elif model_version >= 10 and model_version <= 19:
    elif model_version == 10:
        print(" - rTDOA w/ HiFTA depth w/ LMFB %d model is selected ..." % model_parameters['hifta_depth'])
        model = RobustTDoA_HiFTA(
            stft_size=(model_parameters['stft_h_size'], model_parameters['stft_w_size']),
            stft_ch=2 * 2,  # Audio ch. * (Mag. & Phase spectrums)
            patch_w_num=model_parameters['num_frame'],
            pixel_h_num=model_parameters['pixel_h_num'],
            ###
            patch_dim=model_parameters['patch_dim'],
            pixel_dim=model_parameters['pixel_dim'],
            depth=model_parameters['hifta_depth'],
            heads=model_parameters['hifta_heads'],
            dim_head=model_parameters['hifta_dim_head'],
            delay_len=model_parameters['delay_len'],
            mlp_head_dim=model_parameters['mlp_hidden_dim'],
            ff_dropout=0.1,
            attn_dropout=0.1,
            w_hifta=model_parameters['w_hifta'],
            num_labels=model_parameters['num_TDOA_labels'],
            return_features=model_parameters['return_features'],
            use_mel_filter=True,
            fs=model_parameters['fs']
        )

        # model_name = 'TNT_lmf_depth%d' % model_parameters['tnt_depth']
        model_name = 'HiFTA_LMFB_depth%d' % model_parameters['hifta_depth']

    # elif model_version >= 20 and model_version <= 29:
    elif model_version == 20:
        print(" - rTDOA w/ Transformer w/ LMFB depth %d model is selected ..." % model_parameters['tr_depth'])
        model = RobustTDoA_Trans(
            stft_size=(model_parameters['stft_h_size'], model_parameters['stft_w_size']),
            stft_ch=2 * 2,
            patch_num=(model_parameters['T_patch_h_num'], model_parameters['T_patch_w_num']),
            delay_len=model_parameters['delay_len'],
            depth=model_parameters['tr_depth'],
            heads=model_parameters['tr_heads'],
            dim_head=model_parameters['tr_dim_head'],
            embed_dim=model_parameters['tr_embed_dim'],
            mlp_dim=model_parameters['tr_mlp_dim'],
            mlp_head_dim=model_parameters['mlp_hidden_dim'],
            dropout=model_parameters['tr_dropout'],
            emb_dropout=model_parameters['tr_emb_dropout'],
            num_label=model_parameters['num_TDOA_labels'],
            return_features=model_parameters['return_features'],
            use_mel_filter=True,
            fs=model_parameters['fs']
        )
        model_name = 'TR_LMFB_depth%d' % model_parameters['tr_depth']
    elif model_version == 9:
        model = deep_gcc()
        model_name = 'DeepGCC'

    return model, model_name

def select_model_2nd_stage(model_version, num_mic_pair,
                           model_parameters):
    model_name = None
    if model_version == 1:  # CRNN
        model_name = 'CRNN'
        print(" - CRNN model is selected ...")
    elif model_version == 5:  # CRNN v2
        model_name = 'CRNN_v2'
        model_version = 1
        print(" - CRNN_v2 model is selected ...")
    elif model_version == 2:  # TNT
        model_name = 'HiFTA_depth%d' % model_parameters['depth']
        print(" - HiFTA depth %d model w/o LMFB is selected ..." % model_parameters['depth'])
    elif model_version == 3:  # CRNN w/ learnable mel-scaled filter
        model_name = 'CRNN_LMFB'
        print(" - CRNN_LMFB model is selected ...")
    # elif model_version == 4:  # TNT w/ learnable mel-scaled filter
    #     model_name = 'TNT_lmf'
    #     print(" - TNT_lmf model is selected ...")
    # elif model_version >= 10 and model_version <= 19:   # TNT w/ learnable mel-scaled filter
    elif model_version == 10:  # TNT w/ learnable mel-scaled filter
        model_name = 'HiFTA_LMFB_depth%d' % model_parameters['depth']
        print(" - HiFTA_LMFB depth %d model is selected ..." % model_parameters['depth'])
        model_version = 4
    # elif model_version >= 20 and model_version <= 29:
    elif model_version == 20:
        model_name = 'TR_LMFB_depth%d' % model_parameters['tr_depth']
        print(" - Transformer-LMFB depth %d model is selected ..." % model_parameters['tr_depth'])
        model_version = 5

    model = MLP_DOA(num_pair=num_mic_pair, selected_model=model_version,
                    parameters=model_parameters)

    return model, model_name

### Using TDoA prediction as a input feature, not a TDOA feature
def select_model_v2_2nd_stage(model_version, num_mic_pair,
                           model_parameters):
    model_name = None
    if model_version == 1:  # CRNN
        model_name = 'CRNN'
        print(" - CRNN model is selected ...")
    elif model_version == 5:  # CRNN v2
        model_name = 'CRNN_v2'
        model_version = 1
        print(" - CRNN_v2 model is selected ...")
    elif model_version == 2:  # TNT
        model_name = 'HiFTA_depth%d' % model_parameters['depth']
        print(" - HiFTA depth %d model w/o LMFB is selected ..." % model_parameters['depth'])
    elif model_version == 3:  # CRNN w/ learnable mel-scaled filter
        model_name = 'CRNN_LMFB'
        print(" - CRNN_LMFB model is selected ...")
    # elif model_version == 4:  # TNT w/ learnable mel-scaled filter
    #     model_name = 'TNT_lmf'
    #     print(" - TNT_lmf model is selected ...")
    # elif model_version >= 10 and model_version <= 19:   # TNT w/ learnable mel-scaled filter
    elif model_version == 10:  # TNT w/ learnable mel-scaled filter
        model_name = 'HiFTA_LMFB_depth%d' % model_parameters['depth']
        print(" - HiFTA_LMFB depth %d model is selected ..." % model_parameters['depth'])
        model_version = 4
    # elif model_version >= 20 and model_version <= 29:
    elif model_version == 20:
        model_name = 'TR_LMFB_depth%d' % model_parameters['tr_depth']
        print(" - Transformer-LMFB depth %d model is selected ..." % model_parameters['tr_depth'])
        model_version = 5

    model = MLP_DOA_v2(num_pair=num_mic_pair, selected_model=model_version,
                    parameters=model_parameters)

    return model, model_name


def get_result_folders(project_dir, result_folder, USE_SSLR, USE_DCASE, USE_TUT):

    if USE_SSLR==True and USE_DCASE==True and USE_TUT==True:
        middle_folder_name = 'stft_SSLR_DCASE_TUT'
    elif USE_SSLR==True and USE_DCASE==True and USE_TUT==False:
        middle_folder_name = 'stft_SSLR_DCASE'
    elif USE_SSLR == True and USE_DCASE == False and USE_TUT == False:
        middle_folder_name = 'stft_SSLR'
    elif USE_SSLR == False and USE_DCASE == True and USE_TUT == False:
        middle_folder_name = 'stft_DCASE'
    else:   # Check
        middle_folder_name = 'stft'

    model_path = os.path.join(
        project_dir, result_folder, middle_folder_name, 'models'
    )
    model_path_all_ep = os.path.join(
        project_dir, result_folder, middle_folder_name, 'models', 'epochs'
    )
    log_path = os.path.join(
        project_dir, result_folder, middle_folder_name, 'logs'
    )

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(model_path_all_ep):
        os.makedirs(model_path_all_ep)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    return model_path, model_path_all_ep, log_path

def get_result_directory_1st_stage(model_name, USE_SSLR, USE_DCASE, USE_TUT,
                                   project_dif, result_folder):

    model_path, model_path_all_ep, log_path = get_result_folders(project_dif, result_folder, USE_SSLR, USE_DCASE, USE_TUT)

    model_saved_dir = os.path.join(model_path, 'RobustTDoA_TDOA_DCASE_%r_SSLR_%r_TUT_%r_model_' % (
        USE_DCASE, USE_SSLR, USE_TUT) + model_name + '.pt')
    log_saved_dir = os.path.join(log_path, 'RobustTDoA_TDOA_DCASE_%r_SSLR_%r_TUT_%r_model_' % (
        USE_DCASE, USE_SSLR, USE_TUT) + model_name)
    model_saved_dir_all_ep = os.path.join(model_path_all_ep, 'RobustTDoA_TDOA_DCASE_%r_SSLR_%r_TUT_%r_model_' % (
        USE_DCASE, USE_SSLR, USE_TUT) + model_name)

    return model_saved_dir, model_saved_dir_all_ep, log_saved_dir

def get_result_directory_2nd_stage(dataset_type, model_name, freeze_tdoa_model, USE_SSLR, USE_DCASE, USE_TUT, project_dif, result_folder,
                                   target_ep_TDOA_model=49,
                                   out_dim=3):
    model_path, model_path_all_ep, log_path = get_result_folders(project_dif, result_folder, USE_SSLR, USE_DCASE,
                                                                 USE_TUT)
    _, model_saved_dir_all_ep, _ = get_result_directory_1st_stage(model_name, USE_SSLR, USE_DCASE, USE_TUT,
                                   project_dif, result_folder)
    # 20220613 IK, Modify the saving model folder
    # model_saved_TDOA_dir = model_saved_dir_all_ep + '_ep_%d.pt' % target_ep_TDOA_model
    model_saved_TDOA_dir = model_saved_dir_all_ep + '_ep_%d_flex.pt' % target_ep_TDOA_model

    doa_model_path = os.path.join(model_path, 'tdoa_ep_%d' % target_ep_TDOA_model)
    doa_log_path = os.path.join(log_path, 'tdoa_ep_%d' % target_ep_TDOA_model)

    if not os.path.exists(doa_model_path):
        os.makedirs(doa_model_path)
    if not os.path.exists(doa_log_path):
        os.makedirs(doa_log_path)

    model_saving_DOA_all_ep_dir = None
    log_saved_dir = None

    if dataset_type == 0: # DCASE2021 (Sound Event Localization and Detection, SELD)
        if freeze_tdoa_model:
            model_saving_DOA_all_ep_dir = os.path.join(doa_model_path,
                                                       'RobustTDoA_DoA_DCASE_model_' + model_name + '_d%d_xyz_TDoA_frz' % out_dim)
            log_saved_dir = os.path.join(doa_log_path,
                                         'RobustTDoA_DoA_DCASE_model_' + model_name + '_d%d_xyz_TDoA_frz' % out_dim)


        else:
            model_saving_DOA_all_ep_dir = os.path.join(doa_model_path,
                                                       'RobustTDoA_DoA_DCASE_model_' + model_name + '_d%d_xyz' % out_dim)
            log_saved_dir = os.path.join(doa_log_path,
                                         'RobustTDoA_DoA_DCASE_model_' + model_name + '_d%d_xyz' % out_dim)
    elif dataset_type == 1: # SSLR (Speech-oriented SSL)
        if freeze_tdoa_model:
            model_saving_DOA_all_ep_dir = os.path.join(doa_model_path,
                                                       'RobustTDoA_DoA_SSLR_model_' + model_name + "_TDoA_frz")
            log_saved_dir = os.path.join(doa_log_path,
                                         'RobustTDoA_DoA_SSLR_model_' + model_name + '_TDoA_frz')
        else:
            model_saving_DOA_all_ep_dir = os.path.join(doa_model_path,
                                                       'RobustTDoA_DoA_SSLR_model_' + model_name)
            log_saved_dir = os.path.join(doa_log_path,
                                         'RobustTDoA_DoA_SSLR_model_' + model_name)


    return model_saving_DOA_all_ep_dir, model_saved_TDOA_dir, log_saved_dir