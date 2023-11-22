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
# _GT_FRAME_PATTERN = '%s.fs_%d_hop_len_%d.csv'
_GT_FRAME_PATTERN = '%s.csv'

_VAD_RATE = 100
# _VOICE_THRESHOLD = 1.0    # Prior value
_VOICE_THRESHOLD = 0.5
_SILENCE_THRESHOLD = 0.0

_VOICE_TYPE = 1

PRINT_DEBUG = False

def _file2sid(filepath):
    slpos = filepath.rfind('/') + 1
    dotpos = filepath.rfind('.')
    name = filepath[slpos:dotpos]
    return name

def main(folder, phal_file, params):

    path = params['dataset_dir_sslr']

    outpath = path

    # load phoneme alignment
    phal = np.load(phal_file)

    # fs = 24000
    # label_hop_len_s = 0.1
    fs = params['fs_Est_TDOA']
    abbreiv_fs = fs // 1000  # 16000 --> 16 k
    label_hop_len_s = params['label_hop_len_s_Est_TDOA']
    label_hop_len = int(label_hop_len_s * fs)
    label_frame_res = fs / float(label_hop_len)
    nb_label_frames_1s = int(label_frame_res)

    # frame_size = 10  # 1 second
    frame_size = params['frame_size_Est_TDOA']  # 1 second
    # batch_size = 256
    # batch_size = params['batch_size_Est_TDOA']
    batch_size = 1
    packing_len = label_hop_len * batch_size * frame_size
    packing_fr_len = batch_size * frame_size

    # out_packed_dir = 'packed'
    out_packed_dir = params['folder_for_Est_TDOA']
    # ds_folder = params['folder_for_downsampled_audio']
    _GT_FILE_DIR = 'gt_file'
    _AUDIO_DIR = 'audio_%dk' % abbreiv_fs
    _GT_FRAME_DIR = 'gt_frame_%dk_batch_%d' % (abbreiv_fs, batch_size)
    _AUDIO_OUT_DIR = 'audio_%dk_batch_%d' % (abbreiv_fs, batch_size)

    # for all files
    # audiodir = os.path.join(path, ds_folder, folder, _AUDIO_DIR)
    # gtfiledir = os.path.join(path, ds_folder, folder, _GT_FILE_DIR)
    audiodir = os.path.join(path, folder, _AUDIO_DIR)
    gtfiledir = os.path.join(path, folder, _GT_FILE_DIR)

    gtoutdir = os.path.join(outpath, out_packed_dir, folder, _GT_FRAME_DIR)
    audiooutdir = os.path.join(outpath, out_packed_dir, folder,_AUDIO_OUT_DIR)

    if not os.path.exists(gtoutdir):
        os.makedirs(gtoutdir)
        # os.system("sudo mkdir -p " + gtoutdir)
    if not os.path.exists(audiooutdir):
        os.makedirs(audiooutdir)
        # os.system("sudo mkdir -p " + audiooutdir)

    container_audios = np.empty([4, 0])

    ramained_vads = []
    ramained_event_classes = []
    remained_locs = []
    remained_audio_len = 0

    list_sources = []
    list_audio_len = []

    cnt_packed_files = 0

    for f in os.listdir(gtfiledir):
        if f.endswith(_GT_FILE_SUFFIX):  # If the end of file is '.gt.pkl', do process
            n = f[:-len(_GT_FILE_SUFFIX)]  # Extract the pure file name without '.gt.pkl'
            ### Do process
            # process(gt_file, wav_file, out_file, phal, win_size, hop_size):

            gt_file = os.path.join(gtfiledir, f)
            wav_file = os.path.join(audiodir, n + _WAV_SUFFIX)
            hop_size = label_hop_len

            if PRINT_DEBUG:
                print("Input wav: ", wav_file)

            # parse file-level ground truth
            # with open(gt_file, 'r') as f:
            with open(gt_file, 'rb') as f:
                _, _, _, sources = pickle.load(f)

            # # check wav length
            # fs, nchs, nsamples = wav_load_metadata(wav_file)
            fs, audio_data = load_wav(wav_file)

            list_sources.append(sources)
            # list_sources += sources
            container_audios = np.append(container_audios, np.array(audio_data), axis=-1)
            list_audio_len.append(audio_data.shape[1])

            # if container_audios.shape[1] > packing_len:
            while container_audios.shape[1] > packing_len:
                # if wav_file == '/mnt/data/audio_data/Circular_4ch_SSLR/sslr/lsp_train_106/audio_24k/ssl-data_2017-05-13-16-07-36_10.wav':
                #     print("Debugging")
                # if wav_file == "/mnt/data/audio_data/Circular_4ch_SSLR/sslr/lsp_train_106/audio_24k/ssl-data_2017-05-13-19-06-57_18.wav":
                #     print("")

                # if cnt_packed_files == 1781:
                #     print("Debugging")

                nsamples = container_audios.shape[1]
                # for each source
                vads = []
                locs = []
                speakers = []
                event_classes = []
                begin_audio = remained_audio_len

                ### For remained vads
                len_vads = len([t for t in range(0, nsamples - hop_size, hop_size)])
                # len_vads = (nsamples - hop_size)//hop_size + 1

                for fvad in ramained_vads:
                    for idx in range(fvad.shape[0]):
                        tmp_fvad = np.zeros(len_vads)
                        tmp_fvad[:fvad.shape[1]] = fvad[idx,:]
                        vads.append(list(tmp_fvad))
                for evt_class in ramained_event_classes:
                    for idx in range(len(evt_class)):
                        event_classes.append(int(evt_class[idx]))
                for location in remained_locs:
                    for idx in range(len(location)):
                        locs.append(list(location[idx,:]))

                # ramained_vads.append(vads)
                # ramained_event_classes.append(event_classes)
                # remained_locs.append(locs)
                # print("Debugging")

                for idx, sources in enumerate(list_sources):

                    for loc, src_file, begin, _, _, speaker in sources:
                        locs.append(loc)
                        speakers.append(speaker)

                        # vad per sample
                        vad = np.zeros(nsamples)
                        begin = int(begin * fs)
                        begin += begin_audio
                        ### Get the phoneme of the dry source over all samples (100 Hz)
                        ### (0 ) : silence sample
                        ### (>0) : any phoneme
                        v = phal[_file2sid(src_file)] != 1
                        v = v.repeat(fs / _VAD_RATE)  # Expand samples of 100 Hz to samples of 48 kHz
                        al = min(nsamples - begin, len(v))
                        vad[begin:begin + al] = v[:al]  # If there is any speech, vad[xx]=True. Else, vad[xx]=False ...

                        # vad per frame
                        fvad = [np.mean(vad[t:t + hop_size])
                                for t in range(0, nsamples - hop_size, hop_size)]
                        vads.append(fvad)

                    begin_audio += list_audio_len[idx]


                vads = np.array(vads, ndmin=2)
                locs = np.array(locs, ndmin=2)
                # [frame number (int)], [active class index (int)], [event number index (int)], [azimuth (int)], [elevation (int)]
                # event_classes = np.array([9 if speaker[0] == 'm' else 5 for speaker in speakers])
                event_classes = event_classes + [9 if speaker[0] == 'm' else 5 for speaker in speakers]
                event_classes = np.array(event_classes)

                # if cnt_packed_files == 1781:
                #     print("Debugging")
                # If there is no source ... then break
                last_row = vads.any(axis=1)
                if not any(last_row):
                    container_audios = np.empty((4, 0))
                    remained_audio_len = container_audios.shape[1]

                    ### Plush
                    list_sources.clear()
                    list_audio_len.clear()

                    ramained_vads.clear()
                    ramained_event_classes.clear()
                    remained_locs.clear()
                    break

                # print("Debugging")
                # if cnt_packed_files == 34:
                # print("Debugging")

                packed_audio = container_audios[:, :packing_len]
                container_audios = container_audios[:, packing_len:]
                remained_audio_len = container_audios.shape[1]

                packed_vads = vads[:, :packing_fr_len]
                vads = vads[:, packing_fr_len:]

                #############################################3
                ### Saving
                gtout_file = os.path.join(gtoutdir, "packed_gtf_%d.csv" % cnt_packed_files)
                audioout_file = os.path.join(audiooutdir, "packed_audio_%d.wav" % cnt_packed_files)

                if PRINT_DEBUG:
                    print("  Output gt: ", gtout_file)
                    print("  Output audio: ", audioout_file)
                cnt_packed_files += 1

                ### Save wave
                save_wav(audioout_file, fs, packed_audio)

                ### Save GTs
                gtf_list = []
                for fid in range(packed_vads.shape[1]):
                    # event_num_idx = 0
                    for sid in range(packed_vads.shape[0]):

                        if packed_vads[sid, fid] >= _VOICE_THRESHOLD:
                            gtsrc = [fid, event_classes[sid], sid, locs[sid][0], locs[sid][1], locs[sid][2]]
                            gtf_list.append(gtsrc)
                            # event_num_idx += 1
                with open(gtout_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(gtf_list)
                #############################################3

                last_row = vads.any(axis=1)
                vads = vads[last_row]
                event_classes = event_classes[last_row]
                # if len(event_classes) > 0:  # 2021 12 16 IK
                #     event_classes = event_classes[last_row]
                locs = locs[last_row]

                ### Plush
                list_sources.clear()
                list_audio_len.clear()

                ramained_vads.clear()
                ramained_event_classes.clear()
                remained_locs.clear()

                ramained_vads.append(vads)
                ramained_event_classes.append(event_classes)
                remained_locs.append(locs)

                # last_row = vads.any(axis=1)
                # # If there is no source ... then break
                # if not any(last_row):
                #     container_audios = np.empty((4, 0))
                #     remained_audio_len = container_audios.shape[1]
                #
                #     ### Plush
                #     list_sources.clear()
                #     list_audio_len.clear()
                #
                #     ramained_vads.clear()
                #     ramained_event_classes.clear()
                #     remained_locs.clear()
                #     break
                # else:
                #     vads = vads[last_row]
                #     event_classes = event_classes[last_row]
                #     locs = locs[last_row]
                #
                #     ### Plush
                #     list_sources.clear()
                #     list_audio_len.clear()
                #
                #     ramained_vads.clear()
                #     ramained_event_classes.clear()
                #     remained_locs.clear()
                #
                #     ramained_vads.append(vads)
                #     ramained_event_classes.append(event_classes)
                #     remained_locs.append(locs)

    ### For remained audio
    if container_audios.shape[1] > 0:
        packed_audio = np.zeros((4, packing_len))
        packed_audio[:, :container_audios.shape[1]] = container_audios

        hop_size = label_hop_len

        nsamples = packing_len
        # for each source
        vads = []
        locs = []
        speakers = []
        event_classes = []
        begin_audio = remained_audio_len

        ### For remained vads
        len_vads = len([t for t in range(0, nsamples - hop_size, hop_size)])

        for fvad in ramained_vads:
            for idx in range(fvad.shape[0]):
                tmp_fvad = np.zeros(len_vads)
                tmp_fvad[:fvad.shape[1]] = fvad[idx, :]
                vads.append(list(tmp_fvad))
        for evt_class in ramained_event_classes:
            for idx in range(len(evt_class)):
                event_classes.append(int(evt_class[idx]))
        for location in remained_locs:
            for idx in range(len(location)):
                locs.append(list(location[idx, :]))

        for idx, sources in enumerate(list_sources):

            for loc, src_file, begin, _, _, speaker in sources:
                locs.append(loc)
                speakers.append(speaker)

                # vad per sample
                vad = np.zeros(nsamples)
                begin = int(begin * fs)
                begin += begin_audio
                ### Get the phoneme of the dry source over all samples (100 Hz)
                ### (0 ) : silence sample
                ### (>0) : any phoneme
                v = phal[_file2sid(src_file)] != 1
                v = v.repeat(fs / _VAD_RATE)  # Expand samples of 100 Hz to samples of 48 kHz
                al = min(nsamples - begin, len(v))
                vad[begin:begin + al] = v[:al]  # If there is any speech, vad[xx]=True. Else, vad[xx]=False ...

                # vad per frame
                fvad = [np.mean(vad[t:t + hop_size])
                        for t in range(0, nsamples - hop_size, hop_size)]
                vads.append(fvad)

            begin_audio += list_audio_len[idx]

        vads = np.array(vads, ndmin=2)
        locs = np.array(locs, ndmin=2)
        # [frame number (int)], [active class index (int)], [event number index (int)], [azimuth (int)], [elevation (int)]
        # event_classes = np.array([9 if speaker[0] == 'm' else 5 for speaker in speakers])
        event_classes = event_classes + [9 if speaker[0] == 'm' else 5 for speaker in speakers]
        event_classes = np.array(event_classes)

        packed_vads = np.zeros((vads.shape[0], packing_fr_len))
        packed_vads[:, :vads.shape[1]] = vads
        #############################################3
        ### Saving
        gtout_file = os.path.join(gtoutdir, "packed_gtf_%d.csv" % cnt_packed_files)
        audioout_file = os.path.join(audiooutdir, "packed_audio_%d.wav" % cnt_packed_files)

        if PRINT_DEBUG:
            print("  Output gt: ", gtout_file)
            print("  Output audio: ", audioout_file)
        cnt_packed_files += 1

        ### Save wave
        save_wav(audioout_file, fs, packed_audio)

        ### Save GTs
        gtf_list = []
        for fid in range(packed_vads.shape[1]):
            # event_num_idx = 0
            for sid in range(packed_vads.shape[0]):

                if packed_vads[sid, fid] >= _VOICE_THRESHOLD:
                    gtsrc = [fid, event_classes[sid], sid, locs[sid][0], locs[sid][1], locs[sid][2]]
                    gtf_list.append(gtsrc)
                    # event_num_idx += 1
        with open(gtout_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(gtf_list)
        #############################################3

if __name__ == '__main__':

    print("Execute packing_batch_sslr_data.py ... ")
    # parser = argparse.ArgumentParser(description='Execute MLP_GCC')
    # parser.add_argument('-f', '--fs', metavar='fs', type=int, default=48000)
    # args = parser.parse_args()

    params = parameter.get_params()

    # path = "/mnt/data/audio_data/Circular_4ch_SSLR/sslr"
    # path = "/home/sgvr/inkyu/audio_data/Circular_4ch_SSLR/sslr"
    # path = params['dataset_dir_sslr']
    # outpath = path

    # default_phal = os.path.join(os.path.dirname(sys.argv[0]), 'misc',
    #                             'phone_alignment.npz')
    default_phal = params['phal_dir']

    # _FS = 24

    ### Training data
    # train_folders = ["lsp_train_106", "lsp_train_301"]
    train_folders = params['sslr_folders_train']
    for folder in train_folders:
        # total_path = os.path.join(path, folder)
        main(folder, default_phal, params)

    ### Test data
    # test_folders = ["lsp_test_library", "lsp_test_106"]
    test_folders = params['sslr_folders_test']
    for folder in test_folders:
        # total_path = os.path.join(path, folder)
        main(folder, default_phal, params)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4



