'''
Labeling SSLR dataset, with 4-ch circular-shaped microphone array

Make reference labels of frame, 100 ms
Labels:

[frame number (int)], [active class index (int)], [event number index (int)], [azimuth (int)], [elevation (int)]
10,     1,  0,  -50,  30
11,     1,  0,  -50,  30
11,     1,  1,   10, -20
12,     1,  1,   10, -20
13,     1,  1,   10, -20
13,     4,  2,  -40,   0

Azimuth and elevation angles are given in degrees, rounded to the closest integer value, with azimuth and elevation
being zero at the front, azimuth \pi=[-180, 180], and elevation \theta=[-90, 90].
Note that the azimuth angle is increasing counter-clockwise (\pi=90 at the left).

The 12 target sound classes
    0: alarm
    1: crying baby
    2: crash
    3: barking dog
    4: female scream
    5: female speech
    6: footsteps
    7. knocking on door
    8. male scream
    9. male speech
    10. ringing phone
    11. piano

'''

import sys
import os
import argparse
# import cPickle as pickle
import pickle
import wave
import csv

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from apkit.apkit.basic import load_wav, save_wav
import get_param as parameter

_WAV_SUFFIX = '.wav'
_GT_FILE_SUFFIX = '.gt.pkl'
_GT_FRAME_PATTERN = '%s.fs_%d_hop_len_%d.csv'

_VAD_RATE = 100
_VOICE_THRESHOLD = 1.0
_SILENCE_THRESHOLD = 0.0

_VOICE_TYPE = 1

PRINT_DEBUG = False

def _file2sid(filepath):
    slpos = filepath.rfind('/') + 1
    dotpos = filepath.rfind('.')
    name = filepath[slpos:dotpos]
    return name


def main(folder, params):

    path = params['dataset_dir_dcase']
    outpath = path

    # fs = 24000
    # label_hop_len_s = 0.1
    fs = params['fs_Est_TDOA']
    abbreiv_fs = fs // 1000  # 16000 --> 16 k
    label_hop_len_s = params['label_hop_len_s_Est_TDOA']
    label_hop_len = int(label_hop_len_s * fs)   # 2400
    label_frame_res = fs / float(label_hop_len)
    nb_label_frames_1s = int(label_frame_res)

    # frame_size = 10     # 1 second
    frame_size = params['frame_size_Est_TDOA']  # 1 second
    # batch_size = 256
    # batch_size = params['batch_size_Est_TDOA']
    batch_size = 1
    packing_len = label_hop_len * batch_size * frame_size
    packing_fr_len = batch_size * frame_size

    # ds_folder = params['folder_for_downsampled_audio']

    gt_sub_dir = params['dcase_folders_gt']
    audio_sub_dir = params['dcase_folders_audio'] + "_%dk" % abbreiv_fs
    out_packed_dir = params['folder_for_Est_TDOA']
    out_folder = folder + params['dcase_indicate_for_Est_TDOA'] + '_batch_%d' % batch_size

    ###
    # gt_input_dir = os.path.join(path, ds_folder, gt_sub_dir, folder)
    # audio_input_dir = os.path.join(path, ds_folder, audio_sub_dir, folder)
    gt_input_dir = os.path.join(path, gt_sub_dir, folder)
    audio_input_dir = os.path.join(path, audio_sub_dir, folder)

    gt_output_dir = os.path.join(
        outpath, out_packed_dir, gt_sub_dir + "_%dk" % abbreiv_fs, out_folder)
    audio_output_dir = os.path.join(outpath, out_packed_dir, audio_sub_dir, out_folder)


    if not os.path.exists(gt_output_dir):
        os.makedirs(gt_output_dir)
        # os.system("sudo mkdir -p " + gt_output_dir)

    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)
        # os.system("sudo mkdir -p " + audio_output_dir)

    container_audios = np.empty([4, 0])
    container_gts = []

    begin_gt = 0
    max_event_num_idx = 0

    cnt_packed_files = 0

    for f in os.listdir(gt_input_dir):
        unique_name = f[:-len('.csv')]

        gt_input_file = os.path.join(gt_input_dir, f)
        audio_input_file = os.path.join(audio_input_dir,
                                        unique_name + '.wav')
        if PRINT_DEBUG:
            print("Input wav: ", audio_input_file)

        ### Read audio
        fs, audio_data = load_wav(audio_input_file)
        container_audios = np.append(container_audios, np.array(audio_data), axis=-1)

        ### Read CSV (gt) file
        with open(gt_input_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            tmp_gts = list(reader)
            gts_sph = []
            for gt in tmp_gts:
                gts_sph.append(list(map(int, gt)))

        gts_xyz = []
        for gt in gts_sph:
            ele_rad = gt[4] * np.pi / 180.
            azi_rad = gt[3] * np.pi / 180
            tmp_label = np.cos(ele_rad)
            x = np.cos(azi_rad) * tmp_label
            y = np.sin(azi_rad) * tmp_label
            z = np.sin(ele_rad)
            gts_xyz.append([gt[0] + begin_gt, gt[1], gt[2] + max_event_num_idx, x, y, z])

        container_gts += gts_xyz

        if container_audios.shape[1] > packing_len:
            while container_audios.shape[1] > packing_len:

                ### Audio
                packed_audio = container_audios[:, :packing_len]
                container_audios = container_audios[:, packing_len:]

                ### GT
                packed_gts = []
                tmp_container_gts = container_gts.copy()
                container_gts.clear()
                for gt in tmp_container_gts:
                    if gt[0] < packing_fr_len:
                        packed_gts.append(gt)
                    else:
                        container_gts.append([gt[0]-packing_fr_len, gt[1], gt[2], gt[3], gt[4], gt[5]])

                if len(container_gts) > 0:
                    min_event_num_idx = min([gt[2] for gt in container_gts])
                    for idx in range(len(container_gts)):
                        container_gts[idx][2] -= min_event_num_idx
                    max_event_num_idx = max([gt[2] for gt in container_gts])
                else:
                    max_event_num_idx = 0

                #########################################3
                ### Check errors


                #########################################3

                ### Save
                gt_out_file = os.path.join(gt_output_dir, "packed_gtf_%d.csv" % cnt_packed_files)
                audio_out_file = os.path.join(audio_output_dir, "packed_audio_%d.wav" % cnt_packed_files)

                if PRINT_DEBUG:
                    print("  Output gt: ", gt_out_file)
                    print("  Output audio: ", audio_out_file)

                save_wav(audio_out_file, fs, packed_audio)

                with open(gt_out_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(packed_gts)

                cnt_packed_files += 1

            if container_audios.shape[1] > 0:
                begin_gt = container_audios.shape[1] // label_hop_len
                # print("")

        else:
            max_event_num_idx = max([gt[2] for gt in container_gts])
            begin_gt += 600


        # print("Debugging")

        # dir_outfile = os.path.join(outdir, f)
        # print("Output: ", dir_outfile)
        # with open(dir_outfile, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #
        #     writer.writerows(gts_xyz)

    ### Save remaining audio with zero padding
    if container_audios.shape[1] > 0:
        packed_audio = np.zeros((4, packing_len))
        packed_audio[:, :container_audios.shape[1]] = container_audios
        packed_gts = container_gts

        ### Save
        gt_out_file = os.path.join(gt_output_dir, "packed_gtf_%d.csv" % cnt_packed_files)
        audio_out_file = os.path.join(audio_output_dir, "packed_audio_%d.wav" % cnt_packed_files)

        if PRINT_DEBUG:
            print("  Output gt: ", gt_out_file)
            print("  Output audio: ", audio_out_file)

        save_wav(audio_out_file, fs, packed_audio)

        with open(gt_out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(packed_gts)


if __name__ == '__main__':

    print("Execute packing_batch_dcase2021_data.py ... ")
    # parser = argparse.ArgumentParser(description='Execute MLP_GCC')
    # parser.add_argument('-f', '--fs', metavar='fs', type=int, default=48000)
    # args = parser.parse_args()

    params = parameter.get_params()

    # path = "/mnt/data/audio_data/Tetrahedral_4ch_DCASE2021"
    # path = "/home/sgvr/inkyu/audio_data/Tetrahedral_4ch_DCASE2021"
    # path = params['dataset_dir_dcase']
    # outpath = path

    _FS = 24

    ### Training data
    # folders = ["dev-test", "dev-train", "dev-val"]
    folders = params['dcase_folders_train'] + params['dcase_folders_test']
    for folder in folders:
        main(folder, params)


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4



