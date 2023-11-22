import os
import sys
import numpy as np
import pickle

SSLR_DATASET = 0
DCASE2021_DATASET = 1
TUT_CA_DATASET = 2

IS_TRAIN = 0
IS_TEST = 1
#IS_VAL = 2

def get_params():
    # print("Get Default Parameters.")
    params = dict(
                
        ### SSLR dataset
        # db_dir_sslr='/Data/db/sslr',        
        dataset_dir_sslr='/home/ikan/db/sslr',        
        # sslr_human_folders_test=['human'],

        ### DCASE dataset
        # db_dir_dcase='/Data/db/dcase',
        dataset_dir_dcase='/home/ikan/db/dcase',      
        dcase_indicate_for_Est_TDOA='_xyz',

        dcase_folders_gt='metadata_dev',
        dcase_folders_audio='mic_dev',

        ### TUT Circular Array dataset
        # db_dir_tut_ca='/Data/db/tut-ca',
        dataset_dir_tut_ca='/home/ikan/db/tut-ca',      

        tut_ca_subfolders=['CANSYN', 'CRESYN'],
        tut_ca_folders_train=['ov1_split1', 'ov1_split2',
                              'ov2_split1', 'ov2_split2',
                              'ov3_split1', 'ov3_split2'],
        tut_ca_folders_test=['ov1_split3',
                              'ov2_split3',
                              'ov3_split3'],
        # tut_ca_folders_test=[], # 20230828 IK, Because of "OSError [Errno 5] Input/output error"

        tut_ca_indicator_gt='desc_',
        tut_ca_indicator_audio=['wav_', '_30db'],

        ### Others
        folder_for_downsampled_audio='downsample',
        folder_for_Est_TDOA='packed',

        phal_folder='misc',

        project_dir='/Data/projects/Mic_Pair_Train',
        
        folder_ATA_TDOA_result='Mic_Pair_Train/results',  

        batch_size_Est_TDOA=256,     # Prior 256
        fs_Est_TDOA=24000,
        
        label_hop_len_s_Est_TDOA=0.1,
        frame_size_Est_TDOA=10,

        weight_fs_tdoa_label=1,     # Increasing weight, 20220322
        
        ### For DOA & SELD-NET
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        fs=24000,
        nb_mel_bins=64,
        is_accdoa=True,
        doa_objective='mse',

        dropout_rate=0.05,  # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,  # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],
        # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        rnn_size=[128, 128],  # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],  # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 1000.],  # [sed, doa] weight for scaling the DNN outputs
        nb_epochs=300,  # Train for maximum epochs
        epochs_per_fit=5,  # Number of epochs per fit

        # METRIC PARAMETERS
        lad_doa_thresh=20,
        lad_doa_thresh_2d=20,
        doa_prediction_thresh=0.2,

        ### Microphone array parameters
        # 4-ch Tetrahedral microphone array (DCASE)
        mic_rad_dcase=0.042,  # 0.042 m = 4.2 cm
        mic_positions_sph_dcase=[[45, 35], [-45, -35], [135, -35], [-135, 35]],

        # 4-ch Circular microphone array (SSLR)
        mic_pos_sslr = [
            [-0.0267, 0.0343, 0],
            [-0.0267, -0.0343, 0],
            [0.0313, 0.0343, 0],
            [0.0313, -0.0343, 0]],

        # 8-ch Circular microphone array (TUT-CA)
        mic_rad_tut_ca=0.05,  # 5 cm
        mic_positions_sph_tut_ca=[[0, 0], [45, 0], [90, 0], [135, 0], [180, 0], [225, 0], [270, 0], [315, 0]],

        # 4-ch Circular microphone array (Respeaker v2)
        mic_xy_respeaker = 0.02285 # 2.285 cm
    )

    # Set the delay length of TDOA prediction    
    params['half_delay_len'] = 10

    ### Dataset
    params['sslr_folders_train'] = ['lsp_train_106', 'lsp_train_301']
    params['sslr_folders_test'] = ['lsp_test_106', 'lsp_test_library']
    params['dcase_folders_train'] = ['dev-train', 'dev-val']
    params['dcase_folders_test'] = ['dev-test']
    ###

    params['phal_dir']=os.path.join(params['dataset_dir_sslr'], params['phal_folder'], 'phone_alignment.npz')

    
    params['unique_classes'] = {
        'alarm': 0,
        'baby': 1,
        'crash': 2,
        'dog': 3,
        'female_scream': 4,
        'female_speech': 5,
        'footsteps': 6,
        'knock': 7,
        'male_scream': 8,
        'male_speech': 9,
        'phone': 10,
        'piano': 11  # , 'engine':12, 'fire':13
    }

    # params['pts_3d'] = np.load(os.path.join(params['project_dir'], params['icosphere_vertices_dir'],
    #                                         params['icosphere_vertices_file_name'] % params['icosphere_depth']))

    params['unique_classes_doa'] = dict(
        alarm='0',
        babycry='1',
        crash='2',
        dogbark='3',
        scream='4',
        speech='5',
        footsteps='6',
        doorknock='7',
        phonering='8',
        piano='9'
    )

    params['saved_unique_classes_tdoa'] = dict(
        alarm='0',
        babycry='1',
        crash='2',
        dogbark='3',
        femalescream='4',
        femalespeech='5',
        footsteps='6',
        doorknock='7',
        malescream='8',
        malespeech='9',
        phonering='10',
        piano='11',
        clearthroat='12',
        cough='13',
        doorslam='14',
        drawer='15',
        keyboard='16',
        keydrop='17',
        humanlaughter='18',
        pageturn='19'
    )
    params['unique_classes_tdoa'] = dict(
        alarm='0',
        babycry='1',
        crash='2',
        dogbark='3',
        scream='4',
        speech='5',
        footsteps='6',
        doorknock='7',
        phonering='8',
        piano='9',
        clearthroat='10',
        cough='11',
        doorslam='12',
        drawer='13',
        keyboard='14',
        keydrop='15',
        humanlaughter='16',
        pageturn='17',
    )

    params['gt_dict'] = dict(
        alarm=[[], []],  # 0
        babycry=[[], []],  # 1
        crash=[[], []],  # 2
        dogbark=[[], []],  # 3
        scream=[[], []],  # 4
        speech=[[], []],  # 5
        footsteps=[[], []],  # 6
        doorknock=[[], []],  # 7
        phonering=[[], []],  # 8
        piano=[[], []],  # 9
        clearthroat=[[], []],  # 10
        cough=[[], []],  # 11
        doorslam=[[], []],  # 12
        drawer=[[], []],  # 13
        keyboard=[[], []],  # 14
        keydrop=[[], []],  # 15
        humanlaughter=[[], []],  # 16
        pageturn=[[], []],  # 17
    )

    ### Microphone positions
    # 4-ch Tetrahedral microphone array (DCASE)
    params['mic_pos_dcase'] = []
    for mic_pos in params['mic_positions_sph_dcase']:
        azi_rad = mic_pos[0] * np.pi / 180
        ele_rad = mic_pos[1] * np.pi / 180
        tmp_label = np.cos(ele_rad)
        x = np.cos(azi_rad) * tmp_label
        y = np.sin(azi_rad) * tmp_label
        z = np.sin(ele_rad)
        params['mic_pos_dcase'].append([x * params['mic_rad_dcase'], y * params['mic_rad_dcase'], z * params['mic_rad_dcase']])

    # 4-ch Circular microphone array (SSLR)

    # 8-ch Circular microphone array (TUT-CA)
    params['mic_pos_tut_ca'] = []
    for mic_pos in params['mic_positions_sph_tut_ca']:
        azi_rad = mic_pos[0] * np.pi / 180
        ele_rad = mic_pos[1] * np.pi / 180
        tmp_label = np.cos(ele_rad)
        x = np.cos(azi_rad) * tmp_label
        y = np.sin(azi_rad) * tmp_label
        z = np.sin(ele_rad)
        params['mic_pos_tut_ca'].append([x * params['mic_rad_tut_ca'], y * params['mic_rad_tut_ca'], z * params['mic_rad_tut_ca']])

    # 4-ch Circular microphone array (Respeaker v2)
    params['mic_pos_respeaker'] = [
        # [-params['mic_xy_respeaker'], params['mic_xy_respeaker'], 0],  # 1
        # [-params['mic_xy_respeaker'], -params['mic_xy_respeaker'], 0],  # 2
        # [params['mic_xy_respeaker'], params['mic_xy_respeaker'], 0],  # 3
        # [params['mic_xy_respeaker'], -params['mic_xy_respeaker'], 0]  # 4
        [+params['mic_xy_respeaker'], -params['mic_xy_respeaker'], 0],  # 1
        [+params['mic_xy_respeaker'], +params['mic_xy_respeaker'], 0],  # 2
        [-params['mic_xy_respeaker'], +params['mic_xy_respeaker'], 0],  # 3
        [-params['mic_xy_respeaker'], -params['mic_xy_respeaker'], 0]  # 4
    ]

    # 8-ch Cube-shaped microphone array
    ### Prior version 20230129
    # dist_mic = [0.0725, 0.0885, 0.07]   ## x, y, z
    ### Recent version 20230130
    dist_mic_8ch_cube = [0.045, 0.0325, 0.04]  ## x, y, z
    params['mic_pos_8ch_cube'] = [
        [dist_mic_8ch_cube[0], dist_mic_8ch_cube[1], dist_mic_8ch_cube[2]],
        [dist_mic_8ch_cube[0], -dist_mic_8ch_cube[1], dist_mic_8ch_cube[2]],
        [-dist_mic_8ch_cube[0], -dist_mic_8ch_cube[1], dist_mic_8ch_cube[2]],
        [-dist_mic_8ch_cube[0], dist_mic_8ch_cube[1], dist_mic_8ch_cube[2]],
        [dist_mic_8ch_cube[0], dist_mic_8ch_cube[1], -dist_mic_8ch_cube[2]],
        [dist_mic_8ch_cube[0], -dist_mic_8ch_cube[1], -dist_mic_8ch_cube[2]],
        [-dist_mic_8ch_cube[0], -dist_mic_8ch_cube[1], -dist_mic_8ch_cube[2]],
        [-dist_mic_8ch_cube[0], dist_mic_8ch_cube[1], -dist_mic_8ch_cube[2]]
    ]
    dist_offset_8ch_circle = [0.029, 0.028, 0.062]

    params['mic_pos_8ch_circle'] = [
        [dist_offset_8ch_circle[0], dist_offset_8ch_circle[1], 0],
        [dist_offset_8ch_circle[1], -dist_offset_8ch_circle[0], 0],
        [-dist_offset_8ch_circle[0], -dist_offset_8ch_circle[1], 0],
        [-dist_offset_8ch_circle[1], dist_offset_8ch_circle[0], 0],
        [dist_offset_8ch_circle[2], dist_offset_8ch_circle[0], 0],
        [dist_offset_8ch_circle[0], -dist_offset_8ch_circle[2], 0],
        [-dist_offset_8ch_circle[2], -dist_offset_8ch_circle[0], 0],
        [-dist_offset_8ch_circle[0], dist_offset_8ch_circle[2], 0]
    ]

    # Center pos
    params['mic_center_pos_dcase'] = np.mean(np.asarray(params['mic_pos_dcase']), axis=0)
    params['mic_center_pos_sslr'] = np.mean(np.asarray(params['mic_pos_sslr']), axis=0)
    params['mic_center_pos_tut_ca'] = np.mean(np.asarray(params['mic_pos_tut_ca']), axis=0)
    params['mic_center_pos_respeaker'] = np.mean(np.asarray(params['mic_pos_respeaker']), axis=0)
    params['mic_center_pos_8ch_cube'] = np.mean(np.asarray(params['mic_pos_8ch_cube']), axis=0)
    params['mic_center_pos_8ch_circle'] = np.mean(np.asarray(params['mic_pos_8ch_circle']), axis=0)

    # Pair indice
    num_of_mic_dcase = 4
    num_of_mic_sslr = 4
    # num_of_mic_tut_ca = 8
    num_of_mic_tut_ca = 4
    num_of_mic_respeaker = 4
    num_of_mic_8ch_cube = 8
    params['mic_pair_idx'] = [
        [[mix1, mix2]
         for mix1 in range(num_of_mic_dcase) for mix2 in range(mix1 + 1, num_of_mic_dcase)],
        [[mix1, mix2]
         for mix1 in range(num_of_mic_sslr) for mix2 in range(mix1 + 1, num_of_mic_sslr)],
        [[mix1, mix2]
         for mix1 in range(num_of_mic_tut_ca) for mix2 in range(mix1 + 1, num_of_mic_tut_ca)],
        [[mix1, mix2]
         for mix1 in range(num_of_mic_respeaker) for mix2 in range(mix1 + 1, num_of_mic_respeaker)],
        # [[mix1, mix2]
        #  for mix1 in range(num_of_mic_8ch_cube) for mix2 in range(mix1 + 1, num_of_mic_8ch_cube)]
    ]
    ### For 8-ch
    mic_pair_idx_8ch_cube = []
    half_num_of_mic_8ch_cube = 4
    for mic1 in range(half_num_of_mic_8ch_cube):
        for mic2 in range(mic1 + 1, half_num_of_mic_8ch_cube):
            mic_pair_idx_8ch_cube.append([mic1, mic2])
    for mic1 in range(half_num_of_mic_8ch_cube):
        for mic2 in range(mic1 + 1, half_num_of_mic_8ch_cube):
            mic_pair_idx_8ch_cube.append([mic1 + half_num_of_mic_8ch_cube, mic2 + half_num_of_mic_8ch_cube])
    params['mic_pair_idx'].append(mic_pair_idx_8ch_cube)
    num_of_mic_8ch_circle = 8
    mic_pair_idx_8ch_circle = [[mix1, mix2]
         for mix1 in range(num_of_mic_8ch_circle) for mix2 in range(mix1 + 1, num_of_mic_8ch_circle)]
    params['mic_pair_idx'].append(mic_pair_idx_8ch_circle)


    # Pair position
    params['mic_pair_pos_dcase'] = []
    params['mic_pair_pos_sslr'] = []
    params['mic_pair_pos_respeaker'] = []
    params['mic_pair_pos_8ch_cube'] = []
    params['mic_pair_pos_8ch_circle'] = []

    for idx, p_idx in enumerate(params['mic_pair_idx'][0]):
        params['mic_pair_pos_dcase'].append([params['mic_pos_dcase'][p_idx[0]], params['mic_pos_dcase'][p_idx[1]]])
    for idx, p_idx in enumerate(params['mic_pair_idx'][1]):
        params['mic_pair_pos_sslr'].append([params['mic_pos_sslr'][p_idx[0]], params['mic_pos_sslr'][p_idx[1]]])
    for idx, p_idx in enumerate(params['mic_pair_idx'][3]):
        params['mic_pair_pos_respeaker'].append(
            [params['mic_pos_respeaker'][p_idx[0]], params['mic_pos_respeaker'][p_idx[1]]])
    for idx, p_idx in enumerate(params['mic_pair_idx'][4]):
        params['mic_pair_pos_8ch_cube'].append(
            [params['mic_pos_8ch_cube'][p_idx[0]], params['mic_pos_8ch_cube'][p_idx[1]]])
    for idx, p_idx in enumerate(params['mic_pair_idx'][5]):
        params['mic_pair_pos_8ch_circle'].append(
            [params['mic_pos_8ch_circle'][p_idx[0]], params['mic_pos_8ch_circle'][p_idx[1]]])

    return params

def get_dataset_dir_list(dataset, data_type):
    '''

    Args:
        dataset: SSLR_DATASET (=0), DCASE2021_DATASET (=1), TUT_CA_DATASET (=2)
        data_type: IS_TRAIN or IS_TEST

    Returns:

    '''
    params = get_params()

    fs = params['fs_Est_TDOA']
    abbrev_fs = fs // 1000  # 16000 --> 16 k

    file_batch_size = 1
    out_audioDir, out_gtDir = None, None
    if dataset == SSLR_DATASET:


        dir_sslr = os.path.join(params['dataset_dir_sslr'], params['folder_for_Est_TDOA'])
        dir_sslr_train = params['sslr_folders_train']        
        dir_sslr_test = params['sslr_folders_test']

        if data_type == IS_TRAIN:
            audioDir_sslr_train = [os.path.join(os.path.join(dir_sslr, dir), 'audio_%dk_batch_%d' % (abbrev_fs, file_batch_size)) for
                                   dir in
                                   dir_sslr_train]
            gtDir_sslr_train = [os.path.join(os.path.join(dir_sslr, dir), 'gt_frame_%dk_batch_%d' % (abbrev_fs, file_batch_size)) for
                                dir in
                                dir_sslr_train]
            out_audioDir, out_gtDir = audioDir_sslr_train, gtDir_sslr_train        
        else:   # dataset == IS_TEST
            audioDir_sslr_test = [os.path.join(os.path.join(dir_sslr, dir), 'audio_%dk_batch_%d' % (abbrev_fs, file_batch_size)) for
                                  dir in
                                  dir_sslr_test]
            gtDir_sslr_test = [os.path.join(os.path.join(dir_sslr, dir), 'gt_frame_%dk_batch_%d' % (abbrev_fs, file_batch_size)) for
                               dir in
                               dir_sslr_test]
            out_audioDir, out_gtDir = audioDir_sslr_test, gtDir_sslr_test

    elif dataset == DCASE2021_DATASET:
        dir_dcase2021 = os.path.join(params['dataset_dir_dcase'], params['folder_for_Est_TDOA'])

        if data_type == IS_TRAIN:
            audioDir_decase2021_train = [os.path.join(
                # params['dcase_folders_audio'],
                params['dcase_folders_audio'] + "_%dk" % abbrev_fs,
                folder + params['dcase_indicate_for_Est_TDOA'] + '_batch_%d' % file_batch_size
            ) for folder in params['dcase_folders_train']]
            audioDir_decase2021_train = [os.path.join(dir_dcase2021, tmp_dir) for tmp_dir in audioDir_decase2021_train]

            gtDir_dcase2021_train = [os.path.join(
                # params['dcase_folders_gt'],
                params['dcase_folders_gt'] + "_%dk" % abbrev_fs,
                folder + params['dcase_indicate_for_Est_TDOA'] + '_batch_%d' % file_batch_size
            ) for folder in params['dcase_folders_train']]
            gtDir_dcase2021_train = [os.path.join(dir_dcase2021, tmp_dir) for tmp_dir in gtDir_dcase2021_train]

            out_audioDir, out_gtDir = audioDir_decase2021_train, gtDir_dcase2021_train

        else:  # dataset == IS_TEST
            audioDir_decase2021_test = [os.path.join(
                # params['dcase_folders_audio'],
                params['dcase_folders_audio'] + "_%dk" % abbrev_fs,
                folder + params['dcase_indicate_for_Est_TDOA'] + '_batch_%d' % file_batch_size
            ) for folder in params['dcase_folders_test']]
            audioDir_decase2021_test = [os.path.join(dir_dcase2021, tmp_dir) for tmp_dir in audioDir_decase2021_test]

            gtDir_dcase2021_test = [os.path.join(
                # params['dcase_folders_gt'],
                params['dcase_folders_gt'] + "_%dk" % abbrev_fs,
                folder + params['dcase_indicate_for_Est_TDOA'] + '_batch_%d' % file_batch_size
            ) for folder in params['dcase_folders_test']]
            gtDir_dcase2021_test = [os.path.join(dir_dcase2021, tmp_dir) for tmp_dir in gtDir_dcase2021_test]

            out_audioDir, out_gtDir = audioDir_decase2021_test, gtDir_dcase2021_test

    ################################################################################################################
    elif dataset == TUT_CA_DATASET:
        # fs = params['fs_Est_TDOA']
        # abbrev_fs = fs // 1000  # 16000 --> 16 k
        fs_indicator = '_%dk' % abbrev_fs

        if data_type == IS_TRAIN:
            audioDir_tut_ca_train = [
                # audio dir
                os.path.join(params['dataset_dir_tut_ca'], params['folder_for_Est_TDOA'], subfolder,
                             params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][
                                 1] + fs_indicator)
                for subfolder in params['tut_ca_subfolders'] for split_folder in params['tut_ca_folders_train']
            ]
            gtDir_tut_ca_train = [
                # gt dir
                os.path.join(params['dataset_dir_tut_ca'], params['folder_for_Est_TDOA'], subfolder,
                             params['tut_ca_indicator_gt'] + split_folder + fs_indicator)
                for subfolder in params['tut_ca_subfolders'] for split_folder in params['tut_ca_folders_train']
            ]

            out_audioDir, out_gtDir = audioDir_tut_ca_train, gtDir_tut_ca_train
        
        else:  # dataset == IS_TEST
            audioDir_tut_ca_test = [
                # audio dir
                os.path.join(params['dataset_dir_tut_ca'], params['folder_for_Est_TDOA'], subfolder,
                             params['tut_ca_indicator_audio'][0] + split_folder + params['tut_ca_indicator_audio'][
                                 1] + fs_indicator)
                for subfolder in params['tut_ca_subfolders'] for split_folder in params['tut_ca_folders_test']
            ]
            gtDir_tut_ca_test = [
                # gt dir
                os.path.join(params['dataset_dir_tut_ca'], params['folder_for_Est_TDOA'], subfolder,
                             params['tut_ca_indicator_gt'] + split_folder + fs_indicator)
                for subfolder in params['tut_ca_subfolders'] for split_folder in params['tut_ca_folders_test']
            ]

            out_audioDir, out_gtDir = audioDir_tut_ca_test, gtDir_tut_ca_test

    ################################################################################################################

    return out_audioDir, out_gtDir
