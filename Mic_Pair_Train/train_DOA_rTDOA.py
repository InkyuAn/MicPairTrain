import sys
import os
import numpy as np
import time

import argparse
from torch.utils.data import DataLoader

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchsummary
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import utils.utils

import get_param as parameter
from Metric.evaluation_DOA import evaluation_SSLR as sslr_evaluation_function
from Metric.evaluation_DOA import reshape_3Dto2D, get_SELD_Results, segment_labels, regression_label_format_to_output_format, get_accdoa_labels

def decode_pred_to_xyz(pred, pts_xyz, out_dim, thr):
    b, fr, num_pts, num_labels = pred.shape
    reshaped_pred = pred.transpose(0, 1, 3, 2).reshape(-1, num_pts)
    # constructed_pred = reshaped_pred.reshape(b, fr, num_labels, num_pts).transpose(0, 1, 3, 2)

    max_pred_idx = np.argmax(reshaped_pred, axis=1)
    pred_xyz = np.zeros((b*fr*num_labels, out_dim))
    pred_xyz[np.max(reshaped_pred, axis=1) > thr, :] = pts_xyz[max_pred_idx[np.max(reshaped_pred, axis=1) > thr], :out_dim]

    pred_xyz = pred_xyz.reshape((b, fr, num_labels, out_dim))

    pred_out = np.zeros((b, fr, num_labels*out_dim))
    pred_out[:, :, :num_labels] = pred_xyz[:, :, :, 0]
    pred_out[:, :, num_labels:2 * num_labels] = pred_xyz[:, :, :, 1]
    if out_dim > 2:
        pred_out[:, :, 2 * num_labels:] = pred_xyz[:, :, :, 2]
    return pred_out

def main(USE_SSLR, USE_DCASE, USE_TUT, input_args):

    ### Initialize parameters
    freeze_tdoa_model = input_args['freeze_tdoa']
    w_hifta = input_args['w_hifta']
    use_hybrid = input_args['use_hybrid']

    target_tdoa_model_epoch = input_args['target_tdoa_epoch']

    multi_gpu = input_args['multi_gpu']
    epoch = input_args['epoch']
    batch = input_args['batch']
    model_version = input_args['model_version']
    mlp_hidden_dim = input_args['mlp_hidden_dim']

    ### For HiFTA
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

    print("Perform Array-Geometry-Aware training ...")
    print(" - Args, Batch: ", batch, ", Model_version: ", model_version)
    print(" - Args, (DCASE: ", USE_DCASE, "), (SSLR: ", USE_SSLR, "), (TUT: ", USE_TUT, ")")
    print(" - Args, HiFTA", ", hifta_heads: ", hifta_heads, ", hifta_dim_head: ", hifta_dim_head,
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

    # list_mic_pair = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    num_frame = int(hop_len / hop_label_len_s)

    # Set parameters for STFT, 20220620 IK
    fft_length, hop_length, win_length, audio_length_for_1s = utils.utils.set_audio_parameters(fs)
    stft_w_size, stft_h_size, (stft_out_version_dcase, stft_out_version_sslr) = utils.utils.set_STFT_parameters(1,
                                                                                                                fft_length,
                                                                                                                hop_length,
                                                                                                                audio_length_for_1s,
                                                                                                                model_version,
                                                                                                                num_frame)
    
    dataset_name = 'SSLR_dataset'
    selected_dataset_version = 1  # SSLR, 4-ch Circular on Pepper
    selected_mic_pair_idx = params['mic_pair_idx'][1]
    doa_size = 360
    num_doa_labels = 1
    stft_out_version = stft_out_version_sslr
    
    npair = len(selected_mic_pair_idx)
    doa_thresh = params['lad_doa_thresh']

    print("!!! Selected Dataset: ", dataset_name, " !!!")

    ###  Initialize Datasets  ########################################################

    train_dataloader, test_dataloader = utils.utils.get_all_dataset_2nd_stage(selected_dataset_version,
                                                                              params,
                                                                              stft_out_version, fft_length, hop_length,
                                                                              win_length, batch_size
                                                                              )
    

    ### Model parameters
    stft_ch = 4
    use_learable_mel_filter = True

    if model_version == 1 or model_version == 2 or model_version == 5:
        use_learable_mel_filter = False

    # For Model (HiFTA)
    pixel_h_num = 16

    # For Transformer
    T_patch_h_num = pixel_h_num  # 16
    T_patch_w_num = num_frame  # 10

    if model_version == 1:
        cnn_f_pool_size=(4, 4, 2)
        pads=((1, 0), (1, 1), (1, 0))
        rnn_size=128
        use_middle_fc = True
    elif model_version == 3:
        cnn_f_pool_size = (4, 4, 4)
        pads=((1, 0), (1, 1), (1, 0))
        rnn_size = 128
        use_middle_fc = False
    elif model_version == 5:
        cnn_f_pool_size = (8, 8, 2)
        pads=((1, 0), (1, 1), (0, 0))
        rnn_size = 256
        use_middle_fc = False
    else:
        cnn_f_pool_size = (1, 1, 1)
        pads = ((1, 1), (1, 1), (1, 1))
        rnn_size = 128
        use_middle_fc = False
    tdoa_model_parameters = dict(
        fs=fs,
        mlp_hidden_dim=mlp_hidden_dim,
        cnn_t_pool_size=(2, 2, 1),
        # cnn_f_pool_size=(8, 8, 4),
        rnn_size=rnn_size,
        cnn_f_pool_size=cnn_f_pool_size,
        pads=pads,
        num_labels=len(params['unique_classes_tdoa']),   # 18
        return_features=True,
        use_middle_fc=use_middle_fc,
        delay_len=delay_len,
        ###
        stft_size=(stft_h_size, stft_w_size),
        # stft_ch=2 * 2,  # Audio ch. * (Mag. & Phase spectrums)
        stft_ch=stft_ch,
        patch_w_num=num_frame,
        pixel_h_num=pixel_h_num,
        patch_dim=patch_dim,
        pixel_dim=pixel_dim,
        depth=hifta_depth,
        heads=hifta_heads,
        dim_head=hifta_dim_head,
        ff_dropout=0.1,
        attn_dropout=0.1,
        w_hifta=w_hifta,
        ##############################################################
        tr_patch_num=(T_patch_h_num, T_patch_w_num),
        tr_depth=tr_depth,
        tr_heads=tr_heads,
        tr_dim_head=tr_dim_head,
        tr_embed_dim=tr_embed_dim,
        tr_mlp_dim=tr_mlp_dim,
        tr_dropout=tr_dropout,
        tr_emb_dropout=tr_emb_dropout,
        ##############################################################
        # out_dim=out_dim,  ### 2D or 3D
        # out_size=360 if out_dim == 2 else len(params['pts_3d']),
        # out_size=360,
        out_size=doa_size,
        # num_doa_labels=1,
        num_doa_labels=num_doa_labels,
        use_lmf=use_learable_mel_filter
    )

    model, model_name = utils.utils.select_model_2nd_stage(model_version, npair, tdoa_model_parameters)

    model.eval()
    # torchsummary.summary(model, input_size=(4, 10, 1201))
    pytorch_doa_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(" - Total num. of trainable params in DoA model: ", pytorch_doa_params)
    
    criterion = torch.nn.MSELoss()
    criterion_TDOA = torch.nn.BCELoss()
    criterion_alpha = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model_saving_DOA_all_ep_dir, _, log_saved_dir \
        = utils.utils.get_result_directory_2nd_stage(1 if doa_size == 360 else 0,   # 1: SSLR (Speech-SSL), 0: DCASE (SELD)
                                                     model_name,
                                                     freeze_tdoa_model,
                                                     USE_SSLR,
                                                     USE_DCASE,
                                                     USE_TUT,
                                                     params['project_dir'],
                                                     params['folder_ATA_TDOA_result'],
                                                     target_ep_TDOA_model=target_tdoa_model_epoch
                                                     )
    # log_file = open(log_saved_dir + "._" + dataset_name + ".txt", 'wt')
    writer = SummaryWriter(log_saved_dir + "_" + dataset_name)

    ### Load saved TDOA model
    model_saved_TDOA_dir = '/Data/projects/Mic_Pair_Train/Mic_Pair_Train/results/RobustTDoA_TDOA_DCASE_True_SSLR_True_TUT_True_model_HiFTA_LMFB_depth5_ep_49_flex.pt'
    print(" - Loaded model: ", model_saved_TDOA_dir)
    pretrained_dict = torch.load(model_saved_TDOA_dir)
    model_dict = model.state_dict()

    print("    : Pretrained keys, ", len(pretrained_dict.keys()), "..., target keys, ", len(model_dict.keys()))
    num_updated_keys = 0
    for key in model_dict:
        # tmp_key = key.replace('._TDOA_model', '')
        tmp_key = key.replace('_TDOA_model.', '')
        if tmp_key in pretrained_dict:
            model_dict[key] = pretrained_dict[tmp_key]
            num_updated_keys += 1
    model.load_state_dict(model_dict)
    print("    : Updated keys, ", num_updated_keys)

    if torch.cuda.device_count() > 1 and multi_gpu:
        print(" - Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    ### Freeze TDOA model
    if freeze_tdoa_model:
        model.module.freeze_tdoa_model()
        print("    : Freezing TDOA model")


    ''' Start Iteration '''
    for ep in range(num_epoch):

        print("Start Epoch, ", ep, " ...")

        ''' 
            ****************************************************************
            **********  Train  *********************************************
            ****************************************************************            
        '''
        start_train = time.time()

        model.train()
        loss_list = []

        gt_list = []
        pred_list = []

        evaluator_train = sslr_evaluation_function()

        for tf, gts, gts_like, gts_pair in train_dataloader:

            pred, tdoa_pred, tdoa_feat = model(tf.to(device))
            tdoa_pred = tdoa_pred.permute(0, 2, 3, 1, 4)
        
            loss_doa = criterion(pred, gts_like.to(device))

            if freeze_tdoa_model:
                loss = loss_doa
            else:
                loss_tdoa = criterion_TDOA(tdoa_pred, gts_pair.to(device))
                loss = criterion_alpha*loss_doa + (1-criterion_alpha)*loss_tdoa

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            ### Evaluation            
            gts = gts.reshape(-1, gts.shape[2])
            pred_eval = pred.reshape(-1, pred.shape[2]).to('cpu').detach()

            num_of_gt = gts[:, 0].numpy()
            gt_doas = gts[:, 1:].numpy()

            nFrame = num_of_gt.shape[0]

            evaluator_train.update_seld_scores(nFrame, num_of_gt, gt_doas, pred_eval.numpy())
        
        MAE_train, ACC_train, Prec_train, Reca_train = evaluator_train.compute_seld_scores()

        tr_loss_ep = sum(loss_list) / len(loss_list)
        log_train = "Epoch (train) %d, loss: %f, ...\n" \
                    "    [Train data] MAE %f, ACC %f, Precision %f, Recall %f ... Time: %fs\n" % \
                    (ep, tr_loss_ep,
                        MAE_train, ACC_train, Prec_train, Reca_train,
                        (time.time() - start_train))
        print(log_train)
        writer.add_scalar("loss/train", tr_loss_ep, ep)
        writer.add_scalar("mae/train", MAE_train, ep)
        writer.add_scalar("acc/train", ACC_train, ep)
        writer.add_scalar("precision/train", Prec_train, ep)
        writer.add_scalar("recall/train", Reca_train, ep)
        # log_file.write(log_train)

        ''' 
            ****************************************************************
            **********  Test  **********************************************
            ****************************************************************
        '''
        start_test = time.time()
        model.eval()

        loss_list = []
        evaluator_test = sslr_evaluation_function()

        for tf, gts, gts_like, gts_pair in test_dataloader:
            pred, tdoa_pred, tdoa_feat = model(tf.to(device))

            loss = criterion(pred, gts_like.to(device))
            loss_list.append(loss.item())

            ### Evaluation
            gts = gts.reshape(-1, gts.shape[2])
            pred_eval = pred.reshape(-1, pred.shape[2]).to('cpu').detach()

            num_of_gt = gts[:, 0].numpy()
            gt_doas = gts[:, 1:].numpy()

            nFrame = num_of_gt.shape[0]

            evaluator_test.update_seld_scores(nFrame, num_of_gt, gt_doas, pred_eval.numpy())

        MAE_test, ACC_test, Prec_test, Reca_test = evaluator_test.compute_seld_scores()
        ts_loss_ep = sum(loss_list) / len(loss_list)

        log_test = "    [Test data, loss: %f] MAE %f, ACC %f, Precision %f, Recall %f ... Time: %fs\n" % \
                    (ts_loss_ep, MAE_test, ACC_test, Prec_test, Reca_test, (time.time() - start_test))
        print(log_test)
        writer.add_scalar("loss/test", ts_loss_ep, ep)
        writer.add_scalar("mae/test", MAE_test, ep)
        writer.add_scalar("acc/test", ACC_test, ep)
        writer.add_scalar("precision/test", Prec_test, ep)
        writer.add_scalar("recall/test", Reca_test, ep)
        # log_file.write(log_test)

        ### Saving model per epoch
        tmp_model_name = model_saving_DOA_all_ep_dir + "_" + dataset_name + '_ep_%d_flex.pt' % ep
        # torch.save(model.state_dict(), model_saving_DOA_all_ep_dir + "_" + dataset_name + '_ep_%d.pt' % ep)
        torch.save(model.module.state_dict(), tmp_model_name)
        print("Saved model: ", tmp_model_name)


    log_file.close()



# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_DOA_rTDOA.py')
    parser.add_argument('--use_sslr', metavar='use_sslr', type=utils.utils.str2bool, default=True)
    parser.add_argument('--use_dcase', metavar='use_dcase', type=utils.utils.str2bool, default=True)
    parser.add_argument('--use_tut', metavar='use_tut', type=utils.utils.str2bool, default=True)

    parser.add_argument('--target_tdoa_epoch', metavar='target_tdoa_epoch', type=int, default=49)

    parser.add_argument('--multi_gpu', metavar='multi_gpu', type=utils.utils.str2bool, default=True)
    parser.add_argument('--epoch', metavar='epoch', type=int, default=25)
    parser.add_argument('--batch', metavar='batch', type=int, default=64)

    parser.add_argument('--frz_tdoa', metavar='frz_tdoa', type=utils.utils.str2bool, default=False)
    parser.add_argument('--w_hifta', metavar='w_hifta', type=utils.utils.str2bool, default=True)
    parser.add_argument('--use_hybrid', metavar='use_hybrid', type=utils.utils.str2bool, default=False)

    parser.add_argument('--model_version', metavar='model_version', type=int, default=10)
    '''
    Model version
    1: CRNN (baseline)
    2: HiFTA (baseline)
    9: Deep-GCC (Prior work, baseline)
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
        target_tdoa_epoch=args.target_tdoa_epoch,
        freeze_tdoa=args.frz_tdoa,
        w_hifta=args.w_hifta,
        use_hybrid=args.use_hybrid,
        multi_gpu=args.multi_gpu,
        epoch=args.epoch,
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