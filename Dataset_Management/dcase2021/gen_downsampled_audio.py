import sys
import os
import argparse
# import cPickle as pickle
import pickle
import wave

import numpy as np
import librosa

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from apkit.apkit.basic import load_wav, load_metadata, save_wav, \
                   stft, istft, cola_hamming, cola_rectangle, \
                   freq_upsample, power, power_db, power_avg, power_avg_db, \
                   power_tf, snr, \
                   steering_vector, compute_delay, \
                   mel, mel_inv, mel_freq_fbank_weight, \
                   vad_by_threshold, \
                   cov_matrix, empirical_cov_mat, empirical_cov_mat_by_block

import get_param as parameter

# _FS = 24
PRINT_DEBUG = False

def main(path, out_path, abbreiv_fs):

    target_FS = abbreiv_fs * 1000
    _WAV_SUFFIX = '.wav'

    if not os.path.exists(out_path):
        os.makedirs(out_path)        
        # os.system("sudo mkdir -p " + out_path)

    for f in os.listdir(path):
        if f.endswith(_WAV_SUFFIX): # If the end of file is '.gt.pkl', do process

            audio_name = os.path.join(path, f)  # Wav file name (input)
            down_audio_name = os.path.join(out_path, f)

            if PRINT_DEBUG:
                print("Origin Audio: ", audio_name)
                print("Target Audio: ", down_audio_name)

            fs, audio_data = load_wav(audio_name)
            # downsampled_audio_data = librosa.resample(audio_data, fs, target_FS)
            downsampled_audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=target_FS)

            save_wav(down_audio_name, target_FS, downsampled_audio_data)
            # print("Debugging point")

if __name__ == '__main__':
    print("Execute gen_downsampled_audio.py ... for DCASE 2021")

    parser = argparse.ArgumentParser(description='Execute gen_downsampled_audio.py')
    parser.add_argument('-f', '--fs', metavar='fs', type=int, default=1)
    args = parser.parse_args()

    params = parameter.get_params()

    # path = "/home/sgvr/inkyu/audio_data/Circular_4ch_SSLR/sslr"
    # db_path = params['db_dir_dcase']
    path = params['dataset_dir_dcase']

    if args.fs == 1:    # Default
        fs = params['fs_Est_TDOA']
    else:
        fs = args.fs

    # if args.fs == 24000:
    #     abbreiv_fs = 24
    # elif args.fs == 16000:
    #     abbreiv_fs = 16
    abbreiv_fs = fs // 1000  # 16000 --> 16 k

    audio_sub_dir = params['dcase_folders_audio']
    audio_out_sub_dir = params['dcase_folders_audio'] + "_%dk" % abbreiv_fs

    ### Training data
    # train_folders = ["lsp_train_106", "lsp_train_301"]
    train_folders = params['dcase_folders_train']
    # ds_folder = params['folder_for_downsampled_audio']
    for folder in train_folders:
        total_path = os.path.join(path, audio_sub_dir, folder)
        # total_path = os.path.join(db_path, audio_sub_dir, folder)
        
        total_out_path = os.path.join(path, audio_out_sub_dir, folder)
        main(total_path, total_out_path, abbreiv_fs)

    ### Test data
    # test_folders = ["lsp_test_library", "lsp_test_106"]
    test_folders = params['dcase_folders_test']
    for folder in test_folders:
        total_path = os.path.join(path, audio_sub_dir, folder)
        # total_path = os.path.join(db_path, audio_sub_dir, folder)

        # total_out_path = os.path.join(path, ds_folder, audio_out_sub_dir, folder)
        total_out_path = os.path.join(path, audio_out_sub_dir, folder)
        main(total_path, total_out_path, abbreiv_fs)

