######
### 1st training stage
###     - train DeepGCC model (prior work) to predict TDoA
######

import sys
import os
import numpy as np
import time

import shutil

import argparse
from torch.utils.data import DataLoader

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchsummary
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import get_param as parameter

import utils.utils
from Metric.evaluation_TDOA import evaluation_tdoa_speech_label as evaluation_function

def main(USE_SSLR, USE_DCASE, USE_TUT, input_args):

    eval_train = input_args['eval_train']

    w_hifta = input_args['w_hifta']
    use_hybrid = input_args['use_hybrid']
    multi_gpu = input_args['multi_gpu']

    # For HiFTA
    epoch = input_args['epoch']
    batch = input_args['batch']
    model_version = input_args['model_version']
    mlp_hidden_dim = input_args['mlp_hidden_dim']
    hifta_depth = input_args['hifta_depth']
    hifta_heads = input_args['hifta_heads']
    hifta_dim_head = input_args['hifta_dim_head']
    patch_dim = input_args['hifta_patch_dim']
    pixel_dim = input_args['hifta_pixel_dim']

    ### For Transformer
    tr_depth = input_args['tr_depth']
    tr_heads = input_args['tr_heads']
    tr_dim_head = input_args['tr_dim_head']
    tr_embed_dim = input_args['tr_embed_dim']
    tr_mlp_dim = input_args['tr_mlp_dim']
    tr_dropout = input_args['tr_dropout']
    tr_emb_dropout = input_args['tr_emb_dropout']

    print("Training DeepGCC model to estimate TDoA ...")
    print(" - Args (SSLR: ", USE_SSLR, "), (DCASE: ", USE_DCASE, "), (TUT: ", USE_TUT, "), Batch: ", batch, ", Model_version: ", model_version)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True
    print(" - device:", device)

    # Get parameters
    params = parameter.get_params()

    batch_size = batch

    # Set parameters
    num_worker_dataloader = 8
    num_epoch = epoch
    fs = params['fs_Est_TDOA']

    increasing_weight = params['weight_fs_tdoa_label']  # 24000 Hz * 2 = 48000 Hz
    half_delay_len = int(params['half_delay_len'] * increasing_weight)
    delay_len = half_delay_len * 2 + 1
    hop_len = 1
    hop_label_len_s = 0.1
    hop_label_len = int(hop_label_len_s * fs)

    # print("Debugging_20220323, half_delay_len, delay_len: ", half_delay_len, ", ", delay_len)  # For debugging, 20220323

    list_mic_pair = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    num_frame = int(hop_len / hop_label_len_s)
    # The number of TODA labels
    num_TDOA_labels = len(params['unique_classes_tdoa'])

    # Set parameters for STFT, return the number of samples
    fft_length, hop_length, win_length, audio_length_for_1s = utils.utils.set_audio_parameters(fs)
    stft_w_size, stft_h_size, stft_out_version = utils.utils.set_STFT_parameters(0, fft_length, hop_length,
                                                                                 audio_length_for_1s, model_version,
                                                                                 num_frame)

    
    # For Model (HiFTA)
    pixel_h_num = 16

    # For Transformer
    T_patch_h_num = pixel_h_num # 16
    T_patch_w_num = num_frame   # 10

    ###  Initialize Datasets  ########################################################
    train_dataloader = utils.utils.get_train_dataset_1st_stage(USE_SSLR, USE_DCASE, USE_TUT,
                                  params,
                                  stft_out_version, fft_length, hop_length, win_length, batch_size,
                                  num_worker_dataloader=num_worker_dataloader, sampling_data=False)

    DCASE2021_dataloader_test = utils.utils.get_test_dataset_1st_stage(0,
                                 params,
                                 stft_out_version, fft_length, hop_length, win_length, batch_size,
                                 num_worker_dataloader=num_worker_dataloader,
                                 sampling_data=False
                                 )
    SSLR_dataloader_test = utils.utils.get_test_dataset_1st_stage(1,
                                 params,
                                 stft_out_version, fft_length, hop_length, win_length, batch_size,
                                 num_worker_dataloader=num_worker_dataloader,
                                 sampling_data=False
                                 )
    TUT_dataloader_test = utils.utils.get_test_dataset_1st_stage(2,
                                 params,
                                 stft_out_version, fft_length, hop_length, win_length, batch_size,
                                 num_worker_dataloader=num_worker_dataloader,
                                 sampling_data=False
                                 )

    ###  Initialize Model  ###########################################################
    model_parameters = dict(
        fs=fs,
        return_features=False,
        num_TDOA_labels=num_TDOA_labels,
        delay_len=delay_len,
        mlp_hidden_dim=mlp_hidden_dim,
        stft_h_size=stft_h_size,
        stft_w_size=stft_w_size,
        num_frame=num_frame,
        # HiFTA parameters
        pixel_h_num=pixel_h_num,
        patch_dim=patch_dim,
        pixel_dim=pixel_dim,
        hifta_depth=hifta_depth,
        hifta_heads=hifta_heads,
        hifta_dim_head=hifta_dim_head,
        w_hifta=w_hifta,
        # Tr parameters
        T_patch_h_num=T_patch_h_num,
        T_patch_w_num=T_patch_w_num,
        tr_depth=tr_depth,
        tr_heads=tr_heads,
        tr_dim_head=tr_dim_head,
        tr_embed_dim=tr_embed_dim,
        tr_mlp_dim=tr_mlp_dim,
        tr_dropout=tr_dropout,
        tr_emb_dropout=tr_emb_dropout
    )

    model, model_name = utils.utils.select_model_1st_stage(model_version, model_parameters)

    print(">>> Model name: ", model_name)
    # Generate directories for result files
    model_saved_dir, model_saved_dir_all_ep, log_saved_dir \
        = utils.utils.get_result_directory_1st_stage(model_name,
                                                     USE_SSLR,
                                                     USE_DCASE,
                                                     USE_TUT,
                                                     params['project_dir'],
                                                     # params['folder_result'])
                                                     params['folder_ATA_TDOA_result'])

    ### Load saved TDOA model
    # begin_ep = 0
    begin_ep = input_args['begin_ep']
    if begin_ep > 0:
        # model_dir = model_saved_dir_all_ep + '_ep_%d.pt' % ep
        model_dir = model_saved_dir_all_ep + '_ep_%d_flex.pt' % (begin_ep-1)
        model.load_state_dict(torch.load(model_dir))

    #     shutil.copyfile(log_saved_dir, log_saved_dir + 'bp.txt')
    #     log_file = open(log_saved_dir, 'at')
    # else:
    #     log_file = open(log_saved_dir, 'wt')

    writer = SummaryWriter(log_saved_dir)

    # Set models
    # torchsummary.summary(model, input_size=(4, 10, 1201))
    if torch.cuda.device_count() > 1 and multi_gpu:
        print(" - Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    model.eval()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(" - Total num. of trainable params: ", pytorch_total_params)
    print(" - Model keys which will be updated ... ", len(model.state_dict().keys()))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ''' Start Iteration '''
    # for ep in tqdm(range(begin_ep, num_epoch)):
    for ep in range(begin_ep, num_epoch):

        print("Start Epoch, ", ep, " ...")

        ''' 
            ****************************************************************
            **********  Train  *********************************************
            ****************************************************************            
        '''
        start_train = time.time()

        model.train()
        losses_train = []

        for gcc, gts, gts_xyz, data_dype in train_dataloader:
            # stft = stft.view(stft.shape[1:])
            # gts = gts.view(gts.shape[1:])
            # gts_xyz = gts_xyz.view(gts_xyz.shape[1:])

            gts = gts.to(device, dtype=torch.float)
            gcc = gcc.to(device, dtype=torch.float)

            pred = model(gcc)

            loss = criterion(pred, gts)

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())



            # print("Test 3")
        mean_losses = np.average(np.array(losses_train))
        # print("L2 loss (train): ", mean_losses)
        log_train = "[ep: %d] MSE loss (train): %f ... train time: %f\n" % (ep, mean_losses, (time.time() - start_train))
        print(log_train)
        writer.add_scalar('loss/train', mean_losses, ep)
        # log_file.write(log_train)

        ''' 
            ****************************************************************
            **********  Test  **********************************************
            ****************************************************************
        '''
        start_test = time.time()

        model.eval()

        losses_total_test = []

        if USE_SSLR:
            losses_test = []
            for gcc, gts, gts_xyz, data_dype in SSLR_dataloader_test:

                gts = gts.to(device, dtype=torch.float)
                gcc = gcc.to(device, dtype=torch.float)

                pred = model(gcc)

                loss = criterion(pred, gts)
                losses_test.append(loss.item())
                losses_total_test.append(loss.item())

            mean_losses = 1.0 if len(losses_test) <= 0 else np.average(np.array(losses_test))
            log_test = "[ep: %d] MSE loss (test sslr): %f ...\n" % (ep, mean_losses)
            print(log_test)
            writer.add_scalar('loss/sslr/test', mean_losses, ep)
            # log_file.write(log_test)

        if USE_DCASE:

            losses_test = []
            for gcc, gts, gts_xyz, data_dype in DCASE2021_dataloader_test:
                gts = gts.to(device, dtype=torch.float)
                gcc = gcc.to(device, dtype=torch.float)

                pred = model(gcc)

                loss = criterion(pred, gts)
                losses_test.append(loss.item())
                losses_total_test.append(loss.item())

            mean_losses = 1.0 if len(losses_test) <= 0 else np.average(np.array(losses_test))
            log_test = "[ep: %d] MSE loss (test DCASE): %f ...\n" % (ep, mean_losses)
            print(log_test)
            writer.add_scalar('loss/dcase/test', mean_losses, ep)
            # log_file.write(log_test)

        if USE_TUT:
            losses_test = []
            for gcc, gts, gts_xyz, data_dype in TUT_dataloader_test:
                gts = gts.to(device, dtype=torch.float)
                gcc = gcc.to(device, dtype=torch.float)

                pred = model(gcc)

                loss = criterion(pred, gts)
                losses_test.append(loss.item())
                losses_total_test.append(loss.item())

            mean_losses = 1.0 if len(losses_test) <= 0 else np.average(np.array(losses_test))
            log_test = "[ep: %d] MSE loss (test TUT-CA): %f ...\n" % (ep, mean_losses)
            print(log_test)
            writer.add_scalar('loss/tut/test', mean_losses, ep)
            # log_file.write(log_test)

        mean_total_losses = 1.0 if len(losses_total_test) <= 0 else np.average(np.array(losses_total_test))
        log_test = "  [Total test] MSE loss : %f ...\n" % (mean_total_losses)
        print(log_test)
        writer.add_scalar('loss/test', mean_total_losses, ep)
        # log_file.write(log_test)

        print("    Test computing time:", (time.time() - start_test))

        # Save model per epoch
        torch.save(model.state_dict(), model_saved_dir_all_ep + '_ep_%d.pt' % ep)
        # To have the flexibility to load the model any way ...
        torch.save(model.module.state_dict(), model_saved_dir_all_ep + '_ep_%d_flex.pt' % ep)
        print("Saved model: ", model_saved_dir_all_ep + '_ep_%d_flex.pt' % ep)

    log_file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_TDOA_DeepGCC.py')
    parser.add_argument('--use_sslr', metavar='use_sslr', type=utils.utils.str2bool, default=True)
    parser.add_argument('--use_dcase', metavar='use_dcase', type=utils.utils.str2bool, default=True)
    parser.add_argument('--use_tut', metavar='use_tut', type=utils.utils.str2bool, default=True)

    parser.add_argument('--eval_train', metavar='eval_train', type=utils.utils.str2bool, default=False)
    parser.add_argument('--multi_gpu', metavar='multi_gpu', type=utils.utils.str2bool, default=True)
    parser.add_argument('--epoch', metavar='epoch', type=int, default=50)
    parser.add_argument('--begin_ep', metavar='begin_ep', type=int, default=0)
    # parser.add_argument('--batch', metavar='batch', type=int, default=128)
    parser.add_argument('--batch', metavar='batch', type=int, default=512)

    parser.add_argument('--w_hifta', metavar='w_hifta', type=utils.utils.str2bool, default=True)
    parser.add_argument('--use_hybrid', metavar='use_hybrid', type=utils.utils.str2bool, default=False)

    # parser.add_argument('--model_version', metavar='model_version', type=int, default=10)
    parser.add_argument('--model_version', metavar='model_version', type=int, default=9)
    '''
    Model version
    1: CRNN (baseline)
    2: HiFTA (baseline)
    9: Deep-GCC (Prior work, baseline)
    10: HiFTA-LMFB
    20: Transformer-LMFB
    '''

    parser.add_argument('--mlp_hidden_dim', metavar='mlp_hidden_dim', type=int, default=128)
    parser.add_argument('--hifta_depth', metavar='hifta_depth', type=int, default=1)
    parser.add_argument('--hifta_heads', metavar='hifta_heads', type=int, default=4)
    parser.add_argument('--hifta_dim_head', metavar='hifta_dim_head', type=int, default=128)
    parser.add_argument('--hifta_patch_dim', metavar='hifta_patch_dim', type=int, default=512)
    parser.add_argument('--hifta_pixel_dim', metavar='hifta_pixel_dim', type=int, default=64)

    parser.add_argument('--tr_depth', metavar='tr_depth', type=int, default=1)
    parser.add_argument('--tr_heads', metavar='tr_heads', type=int, default=4)
    parser.add_argument('--tr_dim_head', metavar='tr_dim_head', type=int, default=128)
    parser.add_argument('--tr_embed_dim', metavar='tr_embed_dim', type=int, default=256)
    parser.add_argument('--tr_mlp_dim', metavar='tr_mlp_dim', type=int, default=256)
    parser.add_argument('--tr_dropout', metavar='tr_dropout', type=float, default=0.)
    parser.add_argument('--tr_emb_dropout', metavar='tr_emb_dropout', type=float, default=0.)
    args = parser.parse_args()

    input_args = dict(
        eval_train=args.eval_train,
        w_hifta=args.w_hifta,
        use_hybrid=args.use_hybrid,
        multi_gpu=args.multi_gpu,
        epoch=args.epoch,
        begin_ep=args.begin_ep,
        batch=args.batch,
        model_version=args.model_version,
        mlp_hidden_dim=args.mlp_hidden_dim,
        hifta_depth=args.hifta_depth,
        hifta_heads=args.hifta_heads,
        hifta_dim_head=args.hifta_dim_head,
        hifta_patch_dim=args.hifta_patch_dim,
        hifta_pixel_dim=args.hifta_pixel_dim,
        tr_depth=args.tr_depth,
        tr_heads=args.tr_heads,
        tr_dim_head=args.tr_dim_head,
        tr_embed_dim=args.tr_embed_dim,
        tr_mlp_dim=args.tr_mlp_dim,
        tr_dropout=args.tr_dropout,
        tr_emb_dropout=args.tr_emb_dropout
    )

    try:
        # sys.exit(main(args.use_sslr, args.use_dcase, args.epoch, args.batch, args.model_version))
        sys.exit(main(args.use_sslr, args.use_dcase, args.use_tut, input_args))
    except (ValueError, IOError) as e:
        sys.exit(e)