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

# def main(db_path, path, folder, abbreiv_fs):
def main(path, folder, abbreiv_fs):

    # target_FS = _FS * 1000
    target_FS = abbreiv_fs * 1000
    # for all files
    _AUDIO_DIR = 'audio'
    _WAV_SUFFIX = '.wav'
    _AUDIO_16K_DIR = 'audio_%dk' % abbreiv_fs

    audiodir = os.path.join(path, folder, _AUDIO_DIR)
    # downaudiodir = os.path.join(path, ds_folder, folder, _AUDIO_16K_DIR)
    downaudiodir = os.path.join(path, folder, _AUDIO_16K_DIR)

    if not os.path.exists(downaudiodir):
        os.makedirs(downaudiodir)
        # os.system("sudo mkdir -p " + downaudiodir)

    for f in os.listdir(audiodir):
        if f.endswith(_WAV_SUFFIX): # If the end of file is '.gt.pkl', do process

            audio_name = os.path.join(audiodir, f)  # Wav file name (input)
            down_audio_name = os.path.join(downaudiodir, f)

            if PRINT_DEBUG:
                print("Origin Audio: ", audio_name)
                print("Target Audio: ", down_audio_name)

            fs, audio_data = load_wav(audio_name)
            # downsampled_audio_data = librosa.resample(audio_data, fs, target_FS)
            downsampled_audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=target_FS)

            save_wav(down_audio_name, target_FS, downsampled_audio_data)
            # print("Debugging point")

if __name__ == '__main__':
    print("Execute gen_downsampled_audio.py ... for SSLR dataset")

    parser = argparse.ArgumentParser(description='Execute gen_downsampled_audio.py')
    parser.add_argument('-f', '--fs', metavar='fs', type=int, default=1)
    args = parser.parse_args()

    params = parameter.get_params()

    # path = "/home/sgvr/inkyu/audio_data/Circular_4ch_SSLR/sslr"
    # db_path = params['db_dir_sslr']
    path = params['dataset_dir_sslr']

    if args.fs == 1:    # Default
        fs = params['fs_Est_TDOA']
    else:
        fs = args.fs

    # if args.fs == 24000:
    #     abbreiv_fs = 24
    # elif args.fs == 16000:
    #     abbreiv_fs = 16
    abbreiv_fs = fs // 1000  # 16000 --> 16 k

    ### Training data
    train_folders = params['sslr_folders_train']
    # ds_folder = params['folder_for_downsampled_audio']
    for folder in train_folders:        
        main(path, folder, abbreiv_fs)
        # main(db_path, path, folder, abbreiv_fs)

    ### Test data
    test_folders = params['sslr_folders_test']
    for folder in test_folders:
        main(path, folder, abbreiv_fs)
        # main(db_path, path, folder, abbreiv_fs)

