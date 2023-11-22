import argparse
import sys
import os
import csv

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from apkit.apkit.basic import load_wav, save_wav
import get_param as parameter

PRINT_DEBUG = False

def process(target_fs, input_wav_dir, output_wav_dir):
    if not os.path.exists(output_wav_dir):
        os.makedirs(output_wav_dir)
        # os.system("sudo mkdir -p " + output_wav_dir)


    for f in os.listdir(input_wav_dir):
        if f.endswith('.wav'):

            audio_name = os.path.join(input_wav_dir, f)
            down_audio_name = os.path.join(output_wav_dir, f)

            if PRINT_DEBUG:
                print("Origin audio: ", audio_name)
                print("Target audio: ", down_audio_name)

            fs, audio_data = load_wav(audio_name)
            # down_audio_data = librosa.resample(audio_data, fs, target_fs)
            down_audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=target_fs)

            save_wav(down_audio_name, target_fs, down_audio_data)


def main(params):
    print("Execute gen_downsampled_audio.py ... for 8-ch TUT circular array dataset")

    ### Parameters
    target_fs = params['fs_Est_TDOA']
    abbrev_fs = target_fs // 1000   # 16000 --> 16 k
    target_audio_indicator = '_%dk' % abbrev_fs

    ### Training
    path_train = [
        [
            # Input audio (44.1kHz)
            os.path.join(params['dataset_dir_tut_ca'], subfolder,
                      params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][1]),
            # Output audio (24 kHz)
            # os.path.join(params['dataset_dir_tut_ca'], params['folder_for_downsampled_audio'], subfolder,
            #              params['tut_ca_indicator_audio'][0] + split_folder +
            #              params['tut_ca_indicator_audio'][1] + target_audio_indicator)
            os.path.join(params['dataset_dir_tut_ca'], subfolder,
                         params['tut_ca_indicator_audio'][0] + split_folder +
                         params['tut_ca_indicator_audio'][1] + target_audio_indicator)
         ]
        for subfolder in params['tut_ca_subfolders'] for split_folder in params['tut_ca_folders_train']
    ]

    ### Testing
    path_test = [
        [
            # Input audio (44.1kHz)
            os.path.join(params['dataset_dir_tut_ca'], subfolder,
                      params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][1]),
            # os.path.join(params['dataset_dir_tut_ca'], params['folder_for_downsampled_audio'], subfolder,
            #              params['tut_ca_indicator_audio'][0] + split_folder +
            #              params['tut_ca_indicator_audio'][1] + target_audio_indicator)
            os.path.join(params['dataset_dir_tut_ca'], subfolder,
                         params['tut_ca_indicator_audio'][0] + split_folder +
                         params['tut_ca_indicator_audio'][1] + target_audio_indicator)
         ]
        for subfolder in params['tut_ca_subfolders'] for split_folder in params['tut_ca_folders_test']
    ]

    ### Do process ... for all data
    for dirs in path_train:
        input_wav_dir = dirs[0]
        output_wav_dir = dirs[1]

        process(target_fs, input_wav_dir, output_wav_dir)

    for dirs in path_test:
        input_wav_dir = dirs[0]
        output_wav_dir = dirs[1]

        process(target_fs, input_wav_dir, output_wav_dir)


if __name__ == '__main__':

    params = parameter.get_params()

    main(params)