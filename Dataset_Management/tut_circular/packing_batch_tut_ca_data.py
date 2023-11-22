import sys
import os
import csv

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from apkit.apkit.basic import load_wav, save_wav
import get_param as parameter

'''
The 20 target sound classes
    0: alarm
    1: crying baby
    2: crash
    3: barking dog
    4: female scream
    5: female speech (TUT)
    6: footsteps
    7. knocking on door (TUT)
    8. male scream
    9. male speech (TUT)
    10. ringing phone (TUT)
    11. piano
    
    12. clearthroat (clearing throat) (TUT)
    13. cough (TUT)
    14. doorslam (slamming door) (TUT)
    15. drawer (TUT)
    16. keyboard (keyboard clicks) (TUT)
    17. keys Drop (keys dropped on desk) (TUT)    
    18. Human laughter (TUT)
    19. page turn (paper page turning) (TUT)
   
'''

VOICE_THRESHOLD = 0.5
class_indicators = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop',
                    'knock', 'laughter', 'pageturn', 'phone', 'speech']
class_labels = [12, 13, 14, 15, 16, 17, 7, 18, 19, 10, 5]

audio_ch = 8

PRINT_DEBUG = False

def figure_out_event_class(class_str):

    for tmp_class_indicator, tmp_class_label in zip(class_indicators, class_labels):

        if class_str[:len(tmp_class_indicator)] == tmp_class_indicator:
            return tmp_class_label

def process(fs, input_wav_dir, input_gt_dir, output_wav_dir, output_gt_dir,
            label_hop_len, packing_len, packing_fr_len):

    if not os.path.exists(output_wav_dir):
        os.makedirs(output_wav_dir)
        # os.system("sudo mkdir -p " + output_wav_dir)
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)
        # os.system("sudo mkdir -p " + output_gt_dir)

    hop_size = label_hop_len

    container_audios = np.empty([audio_ch, 0])

    remained_event_detections = []
    remained_event_labels = []
    remained_locs = []
    remained_audio_len = 0

    list_sources = []
    list_audio_len = []

    cnt_packed_files = 0

    for gt_file in os.listdir(input_gt_dir):
        unique_name = gt_file[:-len('.csv')]

        input_wav_path = os.path.join(input_wav_dir, unique_name + '.wav')
        input_gt_path = os.path.join(input_gt_dir, gt_file)

        ### Read audio
        _fs, audio_data = load_wav(input_wav_path)
        assert (fs == _fs), 'Audio sampling frequency is not 24kHz'

        ### Read CSV (gt) file
        # tmp_gts = None
        gts_event = []
        gts_sph = []
        gts_durr = []
        with open(input_gt_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            tmp_gts = list(reader)
            tmp_gts = tmp_gts[1:]
            # tmp_gts = [[float(tmp_value_gt) for tmp_value_gt in tmp_gt[1:]] for tmp_gt in tmp_gts]
            gts_event = [tmp_gt[0] for tmp_gt in tmp_gts]
            gts_durr = [[float(tmp_value_gt) for tmp_value_gt in tmp_gt[1:3]] for tmp_gt in tmp_gts]
            gts_sph = [[float(tmp_value_gt) for tmp_value_gt in tmp_gt[3:]] for tmp_gt in tmp_gts]

        ### Transfor Spherical Coord. to Cartesian Coord.
        sources_info = []
        for gt_event, gt_sph, gt_durr in zip(gts_event, gts_sph, gts_durr):
            ele_rad = gt_sph[0] * np.pi / 180.
            azi_rad = gt_sph[1] * np.pi / 180.
            dist = gt_sph[2]
            x = np.cos(azi_rad) * np.cos(ele_rad) * dist
            y = np.sin(azi_rad) * np.cos(ele_rad) * dist
            z = np.sin(ele_rad) * dist

            gt_label = figure_out_event_class(gt_event)
            # print(gt_event)
            sources_info.append([gt_label, gt_durr[0], gt_durr[1], x, y, z])
            # gts_xyz.append([gt[0] + begin_gt, gt[1], gt[2] + max_event_num_idx, x, y, z])

        # append
        list_sources.append(sources_info)
        container_audios = np.append(container_audios, np.array(audio_data), axis=-1)
        list_audio_len.append(audio_data.shape[1])
        # ### Save audio lengths per packed size
        while container_audios.shape[1] > packing_len:
            nsamples = container_audios.shape[1]

            # for each source
            event_detections = []
            locs = []
            tmp_labels = []
            event_labels = []
            begin_audio = remained_audio_len

            ### For remained vads, read them
            len_vads = len([t for t in range(0, nsamples - hop_size, hop_size)])

            for fed in remained_event_detections:
                for idx in range(fed.shape[0]):
                    tmp_fed = np.zeros(len_vads)
                    tmp_fed[:fed.shape[1]] = fed[idx, :]
                    event_detections.append(list(tmp_fed))
            for evt_label in remained_event_labels:
                for idx in range(len(evt_label)):
                    event_labels.append(int(evt_label[idx]))
            for location in remained_locs:
                for idx in range(len(location)):
                    locs.append(list(location[idx, :]))

            ### Refine source information
            for idx, sources in enumerate(list_sources):
                for gt_label, begin, end, x, y, z in sources:
                    locs.append([x, y, z])
                    tmp_labels.append(gt_label)

                    # Event detection per sample
                    ed = np.zeros(nsamples)
                    begin = int(begin * fs)
                    end = int(end * fs)
                    begin += begin_audio
                    end += begin_audio
                    ed[begin:end] = 1

                    # Event detection per frame
                    fed = [np.mean(ed[t:t + hop_size])
                            for t in range(0, nsamples - hop_size, hop_size)]
                    event_detections.append(fed)
                begin_audio += list_audio_len[idx]

            event_detections = np.array(event_detections, ndmin=2)
            locs = np.array(locs, ndmin=2)
            event_labels = np.array(event_labels + tmp_labels)

            # If there is no source ..., then break
            if not any(event_detections.any(axis=1)):
                container_audios = np.empty((audio_ch, 0))
                remained_audio_len = container_audios.shape[1]

                ### Plush
                list_sources.clear()
                list_audio_len.clear()
                remained_event_detections.clear()
                remained_event_labels.clear()
                remained_locs.clear()
                break

            ### Extract packed audio & labels
            packed_audio = container_audios[:, :packing_len]
            container_audios = container_audios[:, packing_len:]
            remained_audio_len = container_audios.shape[1]

            packed_eds = event_detections[:, :packing_fr_len]
            event_detections = event_detections[:, packing_fr_len:]

            #########################################################################################################
            ### Saving
            out_gt_path = os.path.join(output_gt_dir, 'packed_gtf_%d.csv' % cnt_packed_files)
            out_audio_path = os.path.join(output_wav_dir, 'packed_audio_%d.wav' % cnt_packed_files)

            if PRINT_DEBUG:
                print("  Output gt: ", out_gt_path)
                print("  Output audio: ", out_audio_path)

            cnt_packed_files += 1

            ### Save audio file as '*.wav'
            save_wav(out_audio_path, fs, packed_audio)

            ### Save GTs
            gtf_list = []
            for fid in range(packed_eds.shape[1]):
                for sid in range(packed_eds.shape[0]):
                    if packed_eds[sid, fid] >= VOICE_THRESHOLD:
                        tmp_gt_src = [fid, event_labels[sid], sid, locs[sid][0], locs[sid][1], locs[sid][2]]
                        gtf_list.append(tmp_gt_src)

            with open(out_gt_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(gtf_list)
            #########################################################################################################

            # if out_audio_path == '/mnt/ssd1/data/audio_data/Circular_8ch_TUT/packed/CRESYN/wav_ov1_split1_30db_24k/packed_audio_320.wav':
            #     print("Debugging")
            last_row = event_detections.any(axis=1)
            event_detections = event_detections[last_row]
            event_labels = event_labels[last_row]
            # if len(event_labels) > 0:   # 2021 12 16 IK
            #     event_labels = event_labels[last_row]
            locs = locs[last_row]

            ### Plush
            list_sources.clear()
            list_audio_len.clear()

            remained_locs.clear()
            remained_event_labels.clear()
            remained_event_detections.clear()

            remained_locs.append(locs)
            remained_event_labels.append(event_labels)
            remained_event_detections.append(event_detections)

    ### For remained audio
    if container_audios.shape[1] > 0:
        nsamples = packing_len

        ### For remained event detections
        event_detections = []
        locs = []
        tmp_labels = []
        event_labels = []
        begin_audio = remained_audio_len

        ### For remained vads, read them
        len_vads = len([t for t in range(0, nsamples - hop_size, hop_size)])

        for fed in remained_event_detections:
            for idx in range(fed.shape[0]):
                tmp_fed = np.zeros(len_vads)
                tmp_fed[:fed.shape[1]] = fed[idx, :]
                event_detections.append(list(tmp_fed))
        for evt_label in remained_event_labels:
            for idx in range(len(evt_label)):
                event_labels.append(int(evt_label[idx]))
        for location in remained_locs:
            for idx in range(len(location)):
                locs.append(list(location[idx, :]))

        ### Refine source information
        for idx, sources in enumerate(list_sources):
            for gt_label, begin, end, x, y, z in sources:
                locs.append([x, y, z])
                tmp_labels.append(gt_label)

                # Event detection per sample
                ed = np.zeros(nsamples)
                begin = int(begin * fs)
                end = int(end * fs)
                begin += begin_audio
                end += begin_audio
                ed[begin:end] = 1

                # Event detection per frame
                fed = [np.mean(ed[t:t + hop_size])
                       for t in range(0, nsamples - hop_size, hop_size)]
                event_detections.append(fed)
            begin_audio += list_audio_len[idx]

        event_detections = np.array(event_detections, ndmin=2)
        locs = np.array(locs, ndmin=2)
        event_labels = np.array(event_labels + tmp_labels)

        # If there is no source ..., then break
        if not any(event_detections.any(axis=1)):
            container_audios = np.empty((audio_ch, 0))
            remained_audio_len = container_audios.shape[1]

            ### Plush
            list_sources.clear()
            list_audio_len.clear()
            remained_event_detections.clear()
            remained_event_labels.clear()
            remained_locs.clear()

        ### Extract packed audio & labels
        # packed_audio = container_audios[:, :packing_len]
        # container_audios = container_audios[:, packing_len:]
        # remained_audio_len = container_audios.shape[1]
        packed_audio = np.zeros((audio_ch, packing_len))
        packed_audio[:, :container_audios.shape[1]] = container_audios

        packed_eds = np.zeros((event_detections.shape[0], packing_fr_len))
        packed_eds[:, :event_detections.shape[1]] = event_detections
        # event_detections = event_detections[:, packing_fr_len:]
        #########################################################################################################
        ### Saving
        out_gt_path = os.path.join(output_gt_dir, 'packed_gtf_%d.csv' % cnt_packed_files)
        out_audio_path = os.path.join(output_wav_dir, 'packed_audio_%d.wav' % cnt_packed_files)

        if PRINT_DEBUG:
            print("  Output gt: ", out_gt_path)
            print("  Output audio: ", out_audio_path)

        cnt_packed_files += 1

        ### Save audio file as '*.wav'
        save_wav(out_audio_path, fs, packed_audio)

        ### Save GTs
        gtf_list = []
        for fid in range(packed_eds.shape[1]):
            for sid in range(packed_eds.shape[0]):
                if packed_eds[sid, fid] >= VOICE_THRESHOLD:
                    tmp_gt_src = [fid, event_labels[sid], sid, locs[sid][0], locs[sid][1], locs[sid][2]]
                    gtf_list.append(tmp_gt_src)

        with open(out_gt_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(gtf_list)
        #########################################################################################################

def main(params):

    print("Execute packing_batch_tut_ca_data.py ... ")

    ### Parameters
    fs = params['fs_Est_TDOA']
    abbrev_fs = fs // 1000  # 16000 --> 16 k
    fs_indicator = '_%dk' % abbrev_fs

    label_hop_len_s = params['label_hop_len_s_Est_TDOA']
    label_hop_len = int(label_hop_len_s * fs)
    label_frame_res = fs / float(label_hop_len)
    nb_label_frames_1s = int(label_frame_res)

    frame_size = params['frame_size_Est_TDOA']  # 1 second
    packing_len = label_hop_len * frame_size
    packing_fr_len = frame_size

    ### Training
    path_train = [
        [
            # # Input audio dir
            # os.path.join(params['dataset_dir_tut_ca'], params['folder_for_downsampled_audio'], subfolder,
            #              params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][
            #                  1] + fs_indicator),
            # # Input gt dir
            # os.path.join(params['dataset_dir_tut_ca'], params['folder_for_downsampled_audio'], subfolder, params['tut_ca_indicator_gt'] + split_folder),
            # Input audio dir
            os.path.join(params['dataset_dir_tut_ca'], subfolder,
                         params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][
                             1] + fs_indicator),
            # Input gt dir
            os.path.join(params['dataset_dir_tut_ca'], subfolder,
                         params['tut_ca_indicator_gt'] + split_folder),
            # Output audio dir
            os.path.join(params['dataset_dir_tut_ca'], params['folder_for_Est_TDOA'], subfolder,
                         params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][
                             1] + fs_indicator),
            # Output gt dir
            os.path.join(params['dataset_dir_tut_ca'], params['folder_for_Est_TDOA'], subfolder,
                      params['tut_ca_indicator_gt'] + split_folder + fs_indicator)
         ]
        for subfolder in params['tut_ca_subfolders'] for split_folder in params['tut_ca_folders_train']
    ]

    ### Testing
    path_test = [
        [
            # # Input audio dir
            # os.path.join(params['dataset_dir_tut_ca'], params['folder_for_downsampled_audio'], subfolder,
            #              params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][
            #                  1] + fs_indicator),
            # # Input gt dir
            # os.path.join(params['dataset_dir_tut_ca'], params['folder_for_downsampled_audio'], subfolder, params['tut_ca_indicator_gt'] + split_folder),
            # Input audio dir
            os.path.join(params['dataset_dir_tut_ca'], subfolder,
                         params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][
                             1] + fs_indicator),
            # Input gt dir
            os.path.join(params['dataset_dir_tut_ca'], subfolder,
                         params['tut_ca_indicator_gt'] + split_folder),
            # Output wav dir
            os.path.join(params['dataset_dir_tut_ca'], params['folder_for_Est_TDOA'], subfolder,
                         params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][
                             1] + fs_indicator),
            # Output gt dir
            os.path.join(params['dataset_dir_tut_ca'], params['folder_for_Est_TDOA'], subfolder,
                      params['tut_ca_indicator_gt'] + split_folder + fs_indicator)
         ]
        for subfolder in params['tut_ca_subfolders'] for split_folder in params['tut_ca_folders_test']
    ]

    ### Do process ... for all data
    for dirs in path_train:
        input_wav_dir = dirs[0]
        input_gt_dir = dirs[1]
        output_wav_dir = dirs[2]
        output_gt_dir = dirs[3]

        process(fs, input_wav_dir, input_gt_dir, output_wav_dir, output_gt_dir,
                label_hop_len, packing_len, packing_fr_len)

    for dirs in path_test:
        input_wav_dir = dirs[0]
        input_gt_dir = dirs[1]
        output_wav_dir = dirs[2]
        output_gt_dir = dirs[3]

        process(fs, input_wav_dir, input_gt_dir, output_wav_dir, output_gt_dir,
                label_hop_len, packing_len, packing_fr_len)


if __name__ == '__main__':
    params = parameter.get_params()

    main(params)