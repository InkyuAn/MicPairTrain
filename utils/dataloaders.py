import os
import sys

import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import get_param as parameter

# For the second stage (DOA estimation)
import Dataset_Management.dataloader_batch_audios

def get_datasetloader(dataset_type,
                                 params,
                              stft_out_version, fft_length, hop_length, win_length, batch_size,
                              out_dim = 3,
                              num_worker_dataloader = 8,
                              sampling_data = False):
    out_test_dataloader = None
    train_dataloader = None
    if dataset_type == 0:  # DCASE
        audioDir_dcase2021_train, gtDir_dcase2021_train = parameter.get_dataset_dir_list(parameter.DCASE2021_DATASET,
                                                                                         parameter.IS_TRAIN)
        audioDir_dcase2021_test, gtDir_dcase2021_test = parameter.get_dataset_dir_list(parameter.DCASE2021_DATASET,
                                                                                       parameter.IS_TEST)

        train_dataset_audio = audioDir_dcase2021_train
        train_dataset_label = gtDir_dcase2021_train

        train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio,
                                                                                        train_dataset_label, params,
                                                                                        out_type=stft_out_version,
                                                                                        sampling_data=sampling_data,
                                                                                        out_dim=out_dim,
                                                                                        stft_n_fft=fft_length,
                                                                                        stft_hop_len=hop_length,
                                                                                        stft_win_len=win_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_worker_dataloader, pin_memory=True)

        ''' Load SSLR dataset (Testing) '''
        DCASE2021_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(
            audioDir_dcase2021_test, gtDir_dcase2021_test, params,
            out_type=stft_out_version, sampling_data=sampling_data,
            out_dim=out_dim,
            stft_n_fft=fft_length,
            stft_hop_len=hop_length,
            stft_win_len=win_length)
        DCASE2021_dataloader_test = DataLoader(DCASE2021_dataset_test, batch_size=batch_size, shuffle=False,
                                               num_workers=num_worker_dataloader, pin_memory=True)
        out_test_dataloader = DCASE2021_dataloader_test

    elif dataset_type == 1:  # SSLR
        audioDir_sslr_train, gtDir_sslr_train = parameter.get_dataset_dir_list(parameter.SSLR_DATASET,
                                                                               parameter.IS_TRAIN)
        audioDir_sslr_test, gtDir_sslr_test = parameter.get_dataset_dir_list(parameter.SSLR_DATASET,
                                                                             parameter.IS_TEST)

        train_dataset_audio = audioDir_sslr_train
        train_dataset_label = gtDir_sslr_train

        train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio,
                                                                                        train_dataset_label, params,
                                                                                        out_type=stft_out_version,
                                                                                        sampling_data=sampling_data,
                                                                                        stft_n_fft=fft_length,
                                                                                        stft_hop_len=hop_length,
                                                                                        stft_win_len=win_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_worker_dataloader, pin_memory=True)

        ''' Load SSLR dataset (Testing) '''
        SSLR_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(audioDir_sslr_test,
                                                                                            gtDir_sslr_test, params,
                                                                                            out_type=stft_out_version,
                                                                                            sampling_data=sampling_data,
                                                                                            stft_n_fft=fft_length,
                                                                                            stft_hop_len=hop_length,
                                                                                            stft_win_len=win_length)
        SSLR_dataloader_test = DataLoader(SSLR_dataset_test, batch_size=batch_size, shuffle=False,
                                          num_workers=num_worker_dataloader, pin_memory=True)
        out_test_dataloader = SSLR_dataloader_test
    elif dataset_type == 2:  # TUT
        print("Not used in the second stage ... ")

    elif dataset_type == 3:  # Synthetic dataset (dry signals: Speech)
        ''' For Synthetic dataset '''
        audioDir_synt2102_train, gtDir_synt2102_train = parameter.get_dataset_dir_list(parameter.SYNTHETIC_2102_DATASET,
                                                                                       parameter.IS_TRAIN)

        audioDir_synt2102_test, gtDir_synt2102_test = parameter.get_dataset_dir_list(parameter.SYNTHETIC_2102_DATASET,
                                                                                     parameter.IS_TEST)

        ''' Load Syntetic dataset (Training) '''

        train_dataset_audio = audioDir_synt2102_train
        train_dataset_label = gtDir_synt2102_train

        # print("Debugging")
        train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio,
                                                                                        train_dataset_label, params,
                                                                                        out_type=stft_out_version,
                                                                                        sampling_data=sampling_data,
                                                                                        out_dim=out_dim,
                                                                                        stft_n_fft=fft_length,
                                                                                        stft_hop_len=hop_length,
                                                                                        stft_win_len=win_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_worker_dataloader, pin_memory=True)

        ''' Load Synthetic dataset (for all testing datasets) '''
        Synt2102_dataloader_test_list = []
        for tmp_audioDir_synt2102_test, tmp_gtDir_synt2102_test in zip(audioDir_synt2102_test, gtDir_synt2102_test):
            Synt2102_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(
                                            [tmp_audioDir_synt2102_test],
                                            [tmp_gtDir_synt2102_test],
                                            params,
                                            out_type=stft_out_version,
                                            sampling_data=sampling_data,
                                            out_dim=out_dim,
                                            stft_n_fft=fft_length,
                                            stft_hop_len=hop_length,
                                            stft_win_len=win_length)
            Synt2102_dataloader_test = DataLoader(Synt2102_dataset_test, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_worker_dataloader, pin_memory=True)
            Synt2102_dataloader_test_list.append(Synt2102_dataloader_test)
        ## Only for Synthetic dataset ...
        out_test_dataloader = Synt2102_dataloader_test_list
        # ''' Load Synthetic dataset (Testing v1 direct) '''
        # Synt2102_dataset_test_v1_d = Dataset_Management.dataloader_batch_audios.dataset_batch_audios([audioDir_synt2102_test[0]],
        #                                                                                         [gtDir_synt2102_test[0]],
        #                                                                                         params,
        #                                                                                         out_type=stft_out_version,
        #                                                                                         sampling_data=sampling_data,
        #                                                                                         out_dim=out_dim,
        #                                                                                         stft_n_fft=fft_length,
        #                                                                                         stft_hop_len=hop_length,
        #                                                                                         stft_win_len=win_length)
        # Synt2102_dataloader_test_v1_d = DataLoader(Synt2102_dataset_test_v1_d, batch_size=batch_size, shuffle=False,
        #                                       num_workers=num_worker_dataloader, pin_memory=True)
        #
        # ''' Load Synthetic dataset (Testing v1 direct) '''
        # Synt2102_dataset_test_v2_r = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(
        #     [audioDir_synt2102_test[1]],
        #     [gtDir_synt2102_test[1]],
        #     params,
        #     out_type=stft_out_version,
        #     sampling_data=sampling_data,
        #     out_dim=out_dim,
        #     stft_n_fft=fft_length,
        #     stft_hop_len=hop_length,
        #     stft_win_len=win_length)
        # Synt2102_dataloader_test_v2_r = DataLoader(Synt2102_dataset_test_v2_r, batch_size=batch_size, shuffle=False,
        #                                            num_workers=num_worker_dataloader, pin_memory=True)

        # # Only for Synthetic dataset ...
        # out_test_dataloader = [Synt2102_dataloader_test_v1_d, Synt2102_dataloader_test_v2_r]
    elif dataset_type == 4:  # Synthetic 8ch Cube-shaped microphone array dataset (dry signals: Speech)
        ''' For Synthetic dataset '''
        audioDir_synt2102_8ch_train, gtDir_synt2102_8ch_train = parameter.get_dataset_dir_list(parameter.SYNTHETIC_2102_8chCube_DATASET,
                                                                                       parameter.IS_TRAIN)

        audioDir_synt2102_8ch_test, gtDir_synt2102_8ch_test = parameter.get_dataset_dir_list(parameter.SYNTHETIC_2102_8chCube_DATASET,
                                                                                     parameter.IS_TEST)

        ''' Load Syntetic dataset (Training) '''

        train_dataset_audio = audioDir_synt2102_8ch_train
        train_dataset_label = gtDir_synt2102_8ch_train

        train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio,
                                                                                        train_dataset_label, params,
                                                                                        out_type=stft_out_version,
                                                                                        sampling_data=sampling_data,
                                                                                        out_dim=out_dim,
                                                                                        stft_n_fft=fft_length,
                                                                                        stft_hop_len=hop_length,
                                                                                        stft_win_len=win_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_worker_dataloader, pin_memory=True)

        ''' Load Synthetic dataset (for all testing datasets) '''
        Synt2102_8ch_dataloader_test_list = []
        for tmp_audioDir_synt2102_test, tmp_gtDir_synt2102_test in zip(audioDir_synt2102_8ch_test, gtDir_synt2102_8ch_test):
            Synt2102_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(
                                            [tmp_audioDir_synt2102_test],
                                            [tmp_gtDir_synt2102_test],
                                            params,
                                            out_type=stft_out_version,
                                            sampling_data=sampling_data,
                                            out_dim=out_dim,
                                            stft_n_fft=fft_length,
                                            stft_hop_len=hop_length,
                                            stft_win_len=win_length)
            Synt2102_dataloader_test = DataLoader(Synt2102_dataset_test, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_worker_dataloader, pin_memory=True)
            Synt2102_8ch_dataloader_test_list.append(Synt2102_dataloader_test)
        ## Only for Synthetic dataset ...
        out_test_dataloader = Synt2102_8ch_dataloader_test_list
    ###################################################################
    elif dataset_type == 6:  # Synthetic 8ch Circular-shaped microphone array dataset (dry signals: Speech)
        ''' For Synthetic dataset '''
        audioDir_synt2102_8ch_train, gtDir_synt2102_8ch_train = parameter.get_dataset_dir_list(
            parameter.SYNTHETIC_2102_8chCircle_DATASET,
            parameter.IS_TRAIN)

        audioDir_synt2102_8ch_test, gtDir_synt2102_8ch_test = parameter.get_dataset_dir_list(
            parameter.SYNTHETIC_2102_8chCircle_DATASET,
            parameter.IS_TEST)

        ''' Load Syntetic dataset (Training) '''

        train_dataset_audio = audioDir_synt2102_8ch_train
        train_dataset_label = gtDir_synt2102_8ch_train

        train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio,
                                                                                        train_dataset_label, params,
                                                                                        out_type=stft_out_version,
                                                                                        sampling_data=sampling_data,
                                                                                        out_dim=out_dim,
                                                                                        stft_n_fft=fft_length,
                                                                                        stft_hop_len=hop_length,
                                                                                        stft_win_len=win_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_worker_dataloader, pin_memory=True)

        ''' Load Synthetic dataset (for all testing datasets) '''
        Synt2102_8ch_dataloader_test_list = []
        for tmp_audioDir_synt2102_test, tmp_gtDir_synt2102_test in zip(audioDir_synt2102_8ch_test,
                                                                       gtDir_synt2102_8ch_test):
            Synt2102_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(
                [tmp_audioDir_synt2102_test],
                [tmp_gtDir_synt2102_test],
                params,
                out_type=stft_out_version,
                sampling_data=sampling_data,
                out_dim=out_dim,
                stft_n_fft=fft_length,
                stft_hop_len=hop_length,
                stft_win_len=win_length)
            Synt2102_dataloader_test = DataLoader(Synt2102_dataset_test, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_worker_dataloader, pin_memory=True)
            Synt2102_8ch_dataloader_test_list.append(Synt2102_dataloader_test)
        ## Only for Synthetic dataset ...
        out_test_dataloader = Synt2102_8ch_dataloader_test_list
    ###################################################################
    elif dataset_type == 7:  # In Small room, Synthetic 8ch Circular-shaped microphone array dataset (dry signals: Speech)
        ''' For Synthetic dataset '''

        audioDir_synt2102_sm_room_test, gtDir_synt2102_sm_room_test = parameter.get_dataset_dir_list(
            parameter.SYNTHETIC_SmallRoom_8chCircle_DATASET,
            parameter.IS_TEST)

        train_dataloader = None

        ''' Load Synthetic dataset (for all testing datasets) '''
        Synt_sm_room_8ch_dataloader_test_list = []
        for tmp_audioDir_synt_sm_room_test, tmp_gtDir_synt_sm_room_test in zip(audioDir_synt2102_sm_room_test,
                                                                       gtDir_synt2102_sm_room_test):
            Synt_sm_room_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(
                [tmp_audioDir_synt_sm_room_test],
                [tmp_gtDir_synt_sm_room_test],
                params,
                out_type=stft_out_version,
                sampling_data=sampling_data,
                out_dim=out_dim,
                stft_n_fft=fft_length,
                stft_hop_len=hop_length,
                stft_win_len=win_length)
            Synt_sm_room_dataloader_test = DataLoader(Synt_sm_room_dataset_test, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_worker_dataloader, pin_memory=True)
            Synt_sm_room_8ch_dataloader_test_list.append(Synt_sm_room_dataloader_test)
        ## Only for Synthetic dataset ...
        out_test_dataloader = Synt_sm_room_8ch_dataloader_test_list
    ###################################################################
    elif dataset_type == 5:  # Synthetic SELD dataset (dry signals: NIGENS)
        ''' For Synthetic dataset '''
        audioDir_synt_seld_train, gtDir_synt_seld_train = parameter.get_dataset_dir_list(parameter.SYNTHETIC_2102_SELD_DATASET,
                                                                                       parameter.IS_TRAIN)

        audioDir_synt_seld_test, gtDir_synt_seld_test = parameter.get_dataset_dir_list(parameter.SYNTHETIC_2102_SELD_DATASET,
                                                                                     parameter.IS_TEST)

        ''' Load Syntetic dataset (Training) '''

        train_dataset_audio = audioDir_synt_seld_train
        train_dataset_label = gtDir_synt_seld_train

        # print("Debugging")
        train_dataset = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(train_dataset_audio,
                                                                                        train_dataset_label, params,
                                                                                        out_type=stft_out_version,
                                                                                        sampling_data=sampling_data,
                                                                                        out_dim=out_dim,
                                                                                        stft_n_fft=fft_length,
                                                                                        stft_hop_len=hop_length,
                                                                                        stft_win_len=win_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_worker_dataloader, pin_memory=True)

        ''' Load Synthetic dataset (for all testing datasets) '''
        Synt2102_dataloader_test_list = []
        for tmp_audioDir_synt_seld_test, tmp_gtDir_synt_seld_test in zip(audioDir_synt_seld_test, gtDir_synt_seld_test):
            Synt2102_dataset_test = Dataset_Management.dataloader_batch_audios.dataset_batch_audios(
                                            [tmp_audioDir_synt_seld_test],
                                            [tmp_gtDir_synt_seld_test],
                                            params,
                                            out_type=stft_out_version,
                                            sampling_data=sampling_data,
                                            out_dim=out_dim,
                                            stft_n_fft=fft_length,
                                            stft_hop_len=hop_length,
                                            stft_win_len=win_length)
            Synt2102_dataloader_test = DataLoader(Synt2102_dataset_test, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_worker_dataloader, pin_memory=True)
            Synt2102_dataloader_test_list.append(Synt2102_dataloader_test)
        ## Only for Synthetic dataset ...
        out_test_dataloader = Synt2102_dataloader_test_list
    return train_dataloader, out_test_dataloader