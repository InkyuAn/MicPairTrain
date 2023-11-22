######
### 1st training stage
###     - train Robust-TDoA model (ours) to predict TDoA
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

    print("Training Robust-TDoA model to estimate TDoA ...")
    print(" - Args (SSLR: ", USE_SSLR, "), (DCASE: ", USE_DCASE, "), (TUT: ", USE_TUT, "), Batch: ", batch, ", Model_version: ", model_version)
    print(" - Args, HiFTA", ", heads: ", hifta_heads, ", dim_head: ", hifta_dim_head,
          ", patch_dim: ", patch_dim, ", pixel_dim: ", pixel_dim, ", use_hybrid_input: ", use_hybrid)

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

    '''
        Model version
        
    '''

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

        # shutil.copyfile(log_saved_dir, log_saved_dir + 'bp.txt')
        # log_file = open(log_saved_dir, 'at')
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

    criterion_sed = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    L1Loss = torch.nn.L1Loss()

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
        # evaluator_train = evaluation_function()
        evaluator_train_sslr = evaluation_function(half_delay_len=half_delay_len)
        evaluator_train_dcase = evaluation_function(half_delay_len=half_delay_len)
        evaluator_train_tut = evaluation_function(half_delay_len=half_delay_len)
        L1Loss_sslr_list = []
        L1Loss_dcase_list = []
        L1Loss_tut_list = []

        # model_input = torch.zeros(batch_size, num_pairs, 4, stft_h_size, stft_w_size).to(device)
        for stft, gts, gts_xyz, data_dype in train_dataloader:
            # stft = stft.view(stft.shape[1:])
            # gts = gts.view(gts.shape[1:])
            # gts_xyz = gts_xyz.view(gts_xyz.shape[1:])

            gts = gts.to(device, dtype=torch.float)
            stft = stft.to(device, dtype=torch.float)

            for idx, micIdx in enumerate(list_mic_pair):
                gts_pair = gts[:, :, :, idx, :]
                gts_xyz_pair = gts_xyz[:, :, :, idx, :]

                audio_stft_pair = stft[:, idx]

                pred = model(audio_stft_pair)

                loss = criterion_sed(pred, gts_pair)

                # update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses_train.append(loss.item())

                if eval_train:
                    if len(pred[data_dype == 0]) > 0:  # DCASE
                        tmp_L1Loss = L1Loss(pred[data_dype == 0], gts_pair[data_dype == 0])
                        evaluator_train_dcase.update_scores(pred[data_dype == 0].to('cpu').detach().numpy(),
                                                            gts_xyz_pair[data_dype == 0].to('cpu').numpy())
                        L1Loss_dcase_list.append(tmp_L1Loss.item())
                    if len(pred[data_dype == 1]) > 0:  # SSLR
                        tmp_L1Loss = L1Loss(pred[data_dype == 1], gts_pair[data_dype == 1])
                        evaluator_train_sslr.update_scores(pred[data_dype == 1].to('cpu').detach().numpy(),
                                                           gts_xyz_pair[data_dype == 1].to('cpu').numpy())
                        L1Loss_sslr_list.append(tmp_L1Loss.item())
                    if len(pred[data_dype == 2]) > 0:  # TUT
                        tmp_L1Loss = L1Loss(pred[data_dype == 2], gts_pair[data_dype == 2])
                        evaluator_train_tut.update_scores(pred[data_dype == 2].to('cpu').detach().numpy(),
                                                           gts_xyz_pair[data_dype == 2].to('cpu').numpy())
                        L1Loss_tut_list.append(tmp_L1Loss.item())
            # print("Test 3")
        mean_losses = np.average(np.array(losses_train))
    
        writer.add_scalar('loss/train', mean_losses, ep)        
        log_train = "[ep: %d] BCE loss (train): %f ... train time: %f\n" % (ep, mean_losses, (time.time() - start_train))        
        print(log_train)
        # log_file.write(log_train)

        ### Evaluation of Training
        if eval_train:
            tdoa_err_all, peak_err_all, recall_all, precision_all = 0, 0, 0, 0
            total_tdoa_all, L1_all = 0, 0
            ### SSLR
            if USE_SSLR:
                tdoa_err_sslr, peak_err_sslr, recall_sslr, precision_sslr = evaluator_train_sslr.compute_seld_scores()
                total_tdoa_sslr = evaluator_train_sslr.early_stopping_metric(tdoa_err_sslr, peak_err_sslr)

                mean_L1Loss = np.average(L1Loss_sslr_list)

                log_train = "  [SSLR train, L1 %f] TDOA err %f, Peak err %f \n" % \
                           (mean_L1Loss, tdoa_err_sslr, peak_err_sslr)
                print(log_train)
                writer.add_scalar('l1_loss/sslr/train', mean_L1Loss, ep)       
                writer.add_scalar('tdoa_err/sslr/train', tdoa_err_sslr, ep)       
                writer.add_scalar('peak_err/sslr/train', peak_err_sslr, ep)       
                # log_file.write(log_train)

                tdoa_err_all += tdoa_err_sslr
                peak_err_all += peak_err_sslr
                L1_all += mean_L1Loss

            ### DCASE
            if USE_DCASE:
                tdoa_err_dcase, peak_err_dcase, _, _ = evaluator_train_dcase.compute_seld_scores()
                # total_tdoa_dcase = evaluator_train_dcase.early_stopping_metric(tdoa_err_dcase, peak_err_dcase)

                mean_L1Loss = np.average(L1Loss_dcase_list)

                log_train = "  [DCASE train, L1 %f] TDOA err %f, Peak err %f \n" % \
                           (mean_L1Loss, tdoa_err_dcase, peak_err_dcase)
                print(log_train)
                writer.add_scalar('l1_loss/dcase/train', mean_L1Loss, ep)       
                writer.add_scalar('tdoa_err/dcase/train', tdoa_err_dcase, ep)       
                writer.add_scalar('peak_err/dcase/train', peak_err_dcase, ep)       
                # log_file.write(log_train)

                tdoa_err_all += tdoa_err_dcase
                peak_err_all += peak_err_dcase
                L1_all += mean_L1Loss

            ### TUT
            if USE_TUT:
                tdoa_err_tut, peak_err_tut, _, _ = evaluator_train_tut.compute_seld_scores()
                # total_tdoa_tut = evaluator_train_tut.early_stopping_metric(tdoa_err_tut, peak_err_tut)

                mean_L1Loss = np.average(L1Loss_tut_list)

                log_train = "  [TUT train, L1 %f] TDOA err %f, Peak err %f \n" % \
                            (mean_L1Loss, tdoa_err_tut, peak_err_tut)
                print(log_train)
                writer.add_scalar('l1_loss/tut/train', mean_L1Loss, ep)       
                writer.add_scalar('tdoa_err/tut/train', tdoa_err_tut, ep)       
                writer.add_scalar('peak_err/tut/train', peak_err_tut, ep)      
                # log_file.write(log_train)

                tdoa_err_all += tdoa_err_tut
                peak_err_all += peak_err_tut                
                L1_all += mean_L1Loss

            ### Total
            denom_for_deviding = 1
            if USE_DCASE and USE_SSLR and USE_TUT:
                denom_for_deviding = 3
            elif USE_DCASE and USE_SSLR:
                denom_for_deviding = 2

            tdoa_err_all /= denom_for_deviding
            peak_err_all /= denom_for_deviding
            L1_all /= denom_for_deviding

            log_train = "    [Total train, L1 %f] TDOA err %f, Peak err %f\n" % \
                           (L1_all, tdoa_err_all, peak_err_all)
            print(log_train)
            writer.add_scalar('l1_loss/train', L1_all, ep)       
            writer.add_scalar('tdoa_err/train', tdoa_err_all, ep)       
            writer.add_scalar('peak_err/train', peak_err_all, ep)   
            # log_file.write(log_train)

        ''' 
            ****************************************************************
            **********  Test  **********************************************
            ****************************************************************
        '''
        start_test = time.time()

        model.eval()

        tdoa_err_all, peak_err_all = 0, 0
        # total_tdoa_all = 0
        L1_all = 0

        if USE_SSLR:
            evaluator_sslr = evaluation_function(half_delay_len=half_delay_len)
            L1Loss_sslr_list = []

            for stft, gts, gts_xyz, data_dype in (SSLR_dataloader_test):

                gts = gts.to(device, dtype=torch.float)
                stft = stft.to(device, dtype=torch.float)

                for idx, micIdx in enumerate(list_mic_pair):
                    gts_pair = gts[:, :, :, idx, :]
                    gts_xyz_pair = gts_xyz[:, :, :, idx, :]

                    audio_stft_pair = stft[:, idx]
                    pred = model(audio_stft_pair)

                    tmp_L1Loss = L1Loss(pred, gts_pair)
                    L1Loss_sslr_list.append(tmp_L1Loss.item())

                    evaluator_sslr.update_scores(pred.to('cpu').detach().numpy(), gts_xyz_pair.to('cpu').numpy())

            tdoa_err_sslr, peak_err_sslr, _, _ = evaluator_sslr.compute_seld_scores()
            # total_tdoa_sslr = evaluator_sslr.early_stopping_metric(tdoa_err_sslr, peak_err_sslr)

            mean_L1Loss = np.average(L1Loss_sslr_list)

            log_test = "  [SSLR test, L1 %f] TDOA err %f, Peak err %f \n" % \
                       (mean_L1Loss, tdoa_err_sslr, peak_err_sslr)
            print(log_test)
            writer.add_scalar('l1_loss/sslr/test', mean_L1Loss, ep)       
            writer.add_scalar('tdoa_err/sslr/test', tdoa_err_sslr, ep)       
            writer.add_scalar('peak_err/sslr/test', peak_err_sslr, ep)   
            # log_file.write(log_test)

            tdoa_err_all += tdoa_err_sslr
            peak_err_all += peak_err_sslr            
            L1_all += mean_L1Loss

        if USE_DCASE:
            evaluator_dcase = evaluation_function(half_delay_len=half_delay_len)
            L1Loss_dcase_list = []

            for stft, gts, gts_xyz, data_dype in DCASE2021_dataloader_test:
                gts = gts.to(device, dtype=torch.float)
                stft = stft.to(device, dtype=torch.float)

                for idx, micIdx in enumerate(list_mic_pair):
                    gts_pair = gts[:, :, :, idx, :]
                    gts_xyz_pair = gts_xyz[:, :, :, idx, :]

                    audio_stft_pair = stft[:, idx]
                    pred = model(audio_stft_pair)

                    tmp_L1Loss = L1Loss(pred, gts_pair)
                    L1Loss_dcase_list.append(tmp_L1Loss.item())

                    evaluator_dcase.update_scores(pred.to('cpu').detach().numpy(), gts_xyz_pair.to('cpu').numpy())

            tdoa_err_dcase, peak_err_dcase, _, _ = evaluator_dcase.compute_seld_scores()
            # total_tdoa_dcase = evaluator_dcase.early_stopping_metric(tdoa_err_dcase, peak_err_dcase)

            mean_L1Loss = np.average(L1Loss_dcase_list)

            log_test = "  [DCASE test, L1 %f] TDOA err %f, Peak err %f \n" % \
                       (mean_L1Loss, tdoa_err_dcase, peak_err_dcase)
            print(log_test)
            writer.add_scalar('l1_loss/dcase/test', mean_L1Loss, ep)       
            writer.add_scalar('tdoa_err/dcase/test', tdoa_err_dcase, ep)       
            writer.add_scalar('peak_err/dcase/test', peak_err_dcase, ep)   
            # log_file.write(log_test)

            tdoa_err_all += tdoa_err_dcase
            peak_err_all += peak_err_dcase            
            L1_all += mean_L1Loss

        if USE_TUT:
            evaluator_tut = evaluation_function(half_delay_len=half_delay_len)
            L1Loss_tut_list = []

            for stft, gts, gts_xyz, data_dype in TUT_dataloader_test:
                gts = gts.to(device, dtype=torch.float)
                stft = stft.to(device, dtype=torch.float)

                for idx, micIdx in enumerate(list_mic_pair):
                    gts_pair = gts[:, :, :, idx, :]
                    gts_xyz_pair = gts_xyz[:, :, :, idx, :]

                    audio_stft_pair = stft[:, idx]
                    pred = model(audio_stft_pair)

                    tmp_L1Loss = L1Loss(pred, gts_pair)
                    L1Loss_tut_list.append(tmp_L1Loss.item())

                    evaluator_tut.update_scores(pred.to('cpu').detach().numpy(), gts_xyz_pair.to('cpu').numpy())

            tdoa_err_tut, peak_err_tut, _, _ = evaluator_tut.compute_seld_scores()
            # total_tdoa_tut = evaluator_tut.early_stopping_metric(tdoa_err_tut, peak_err_tut)

            mean_L1Loss = np.average(L1Loss_tut_list)

            log_test = "  [TUT test, L1 %f] TDOA err %f, Peak err %f \n" % \
                       (mean_L1Loss, tdoa_err_tut, peak_err_tut)
            print(log_test)
            writer.add_scalar('l1_loss/tut/test', mean_L1Loss, ep)       
            writer.add_scalar('tdoa_err/tut/test', tdoa_err_tut, ep)       
            writer.add_scalar('peak_err/tut/test', peak_err_tut, ep)   
            # log_file.write(log_test)

            tdoa_err_all += tdoa_err_tut
            peak_err_all += peak_err_tut
            L1_all += mean_L1Loss

        denom_for_deviding = 1
        if USE_SSLR and USE_DCASE and USE_TUT:
            denom_for_deviding = 3
        elif USE_SSLR and USE_DCASE:
            denom_for_deviding = 2
        tdoa_err_all /= denom_for_deviding
        peak_err_all /= denom_for_deviding
        L1_all /= denom_for_deviding

        log_test = "    [Total test, L1 %f] TDOA err %f, Peak err %f \n" % \
                  (L1_all, tdoa_err_all, peak_err_all)
        print(log_test)
        writer.add_scalar('l1_loss/test', L1_all, ep)       
        writer.add_scalar('tdoa_err/test', tdoa_err_all, ep)       
        writer.add_scalar('peak_err/test', peak_err_all, ep)  
        # log_file.write(log_test)

        print("    Test computing time:", (time.time() - start_test))

        # Save model per epoch
        torch.save(model.state_dict(), model_saved_dir_all_ep + '_ep_%d.pt' % ep)
        # To have the flexibility to load the model any way ...
        torch.save(model.module.state_dict(), model_saved_dir_all_ep + '_ep_%d_flex.pt' % ep)
        print("Saved model: ", model_saved_dir_all_ep + '_ep_%d.pt' % ep)

    # log_file.close()
    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_TDOA_rTDOA.py')
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
    
    parser.add_argument('--model_version', metavar='model_version', type=int, default=10)
    '''
    Model version
    1: CRNN (baseline)
    2: HiFTA (baseline)
    9: Deep-GCC (Prior work, baseline), Not used here --> Use "train_TDOA_DeepGCC.py"
    10: HiFTA-LMFB
    20: Transformer-LMFB
    '''

    parser.add_argument('--mlp_hidden_dim', metavar='mlp_hidden_dim', type=int, default=128)
    # parser.add_argument('--hifta_depth', metavar='hifta_depth', type=int, default=1)
    parser.add_argument('--hifta_depth', metavar='hifta_depth', type=int, default=5)
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