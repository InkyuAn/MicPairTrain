import torch
import torch.fft
import torch.nn as nn

from .CRNN_stft import CRNNpair
from .Robust_TDoA_HiFTA import RobustTDoA_HiFTA
from .Robust_TDoA_Transformer import RobustTDoA_Trans
import numpy as np

class MLP_DOA_v2(nn.Module):
    def __init__(self, num_pair, selected_model, parameters):
        super(MLP_DOA_v2, self).__init__()

        self._eps = torch.finfo(torch.float).eps

        self._fr_len = 10
        # self.list_mic_pair = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        # self.input_ch = 6 # 4 mics --> 6 pairs
        # self.input_ch = num_pair

        # self._out_dim = parameters['out_dim']

        ### TDOA model
        if selected_model == 1 or selected_model == 3: # CRNN
            self._TDOA_model = CRNNpair(out_size=parameters['delay_len'],
                     cnn_filter_num=parameters['mlp_hidden_dim'], rnn_size=parameters['rnn_size'], fc_size=parameters['mlp_hidden_dim'],
                     cnn_t_pool_size=parameters['cnn_t_pool_size'],
                     cnn_f_pool_size=parameters['cnn_f_pool_size'],
                     pads=parameters['pads'],
                     num_labels=parameters['num_labels'],
                     return_features=parameters['return_features'],
                    stft_h_size=parameters['stft_size'][0],
                    use_mel_filter=parameters['use_lmf'],
                    use_middle_fc=parameters['use_middle_fc'],
                    fs=parameters['fs']
                     )
        # if selected_model == 1:
        #     self._TDOA_model = CRNNpair(out_size=parameters['delay_len'],
        #                      cnn_filter_num=parameters['mlp_hidden_dim'],
        #                      rnn_size=parameters['mlp_hidden_dim'], fc_size=parameters['mlp_hidden_dim'],
        #                      cnn_t_pool_size=(2, 2, 1),
        #                      cnn_f_pool_size=(4, 4, 2),
        #                      # pads=((1, 1), (1, 1), (1, 1)),
        #                      pads=((1, 0), (1, 1), (1, 0)),
        #                      num_labels=parameters['num_labels'],
        #                      return_features=parameters['return_features'],
        #                      fs=parameters['fs']
        #                      )
        # elif selected_model == 3:
        #     ### TODO 20220602 IK, Need to be modified ... CRNN model may contain errors, The CRNN model architecture needs to be modified.
        #     self._TDOA_model = CRNNpair(out_size=parameters['delay_len'],
        #                      cnn_filter_num=parameters['mlp_hidden_dim'],
        #                      rnn_size=parameters['mlp_hidden_dim'], fc_size=parameters['mlp_hidden_dim'],
        #                      cnn_t_pool_size=(2, 2, 1),
        #                      cnn_f_pool_size=(4, 4, 4),
        #                      pads=((1, 0), (1, 1), (1, 0)),
        #                      num_labels=parameters['num_labels'],
        #                      stft_h_size=parameters['stft_size'][0],
        #                      return_features=parameters['return_features'],
        #                      use_mel_filter=True,
        #                      fs=parameters['fs']
        #                      )
        elif selected_model == 2 or selected_model == 4: # HiFTA
            self._TDOA_model = RobustTDoA_HiFTA(
                stft_size=parameters['stft_size'],
                stft_ch=parameters['stft_ch'],
                patch_w_num=parameters['patch_w_num'],
                pixel_h_num=parameters['pixel_h_num'],
                patch_dim=parameters['patch_dim'],
                pixel_dim=parameters['pixel_dim'],
                depth=parameters['depth'],
                heads=parameters['heads'],
                dim_head=parameters['dim_head'],
                delay_len=parameters['delay_len'],
                mlp_head_dim=parameters['mlp_hidden_dim'],
                ff_dropout=parameters['ff_dropout'],
                attn_dropout=parameters['attn_dropout'],
                w_hifta=parameters['w_hifta'],
                num_labels=parameters['num_labels'],
                return_features=parameters['return_features'],
                use_mel_filter=parameters['use_lmf'],
                fs=parameters['fs']
            )
        elif selected_model == 5:   # Transformer
            self._TDOA_model = RobustTDoA_Trans(
                stft_size=parameters['stft_size'],
                stft_ch=parameters['stft_ch'],
                patch_num=parameters['tr_patch_num'],
                delay_len=parameters['delay_len'],
                depth=parameters['tr_depth'],
                heads=parameters['tr_heads'],
                dim_head=parameters['tr_dim_head'],
                embed_dim=parameters['tr_embed_dim'],
                mlp_dim=parameters['tr_mlp_dim'],
                mlp_head_dim=parameters['mlp_hidden_dim'],
                dropout=parameters['tr_dropout'],
                emb_dropout=parameters['tr_emb_dropout'],
                num_label=parameters['num_labels'],
                return_features=parameters['return_features'],
                use_mel_filter=parameters['use_lmf'],
                fs=parameters['fs']
            )

        # if torch.cuda.device_count() > 1 and use_multi_gpu:
        #     # print(" - Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self._TDOA_model = torch.nn.DataParallel(self._TDOA_model)
        # self._TDOA_model.to(device)

        # self.input_features = parameters['mlp_hidden_dim']
        self.input_features = parameters['delay_len'] * parameters['num_labels']
        self.fc_input_size = self.input_features * num_pair
        self.fc_hidden_size = 1000
        # if self._out_dim == 2:
        #     self.fc_out_size = 360
        # elif self._out_dim == 3:
        #     self.fc_out_size = 360
        self._num_doa_labels = parameters['num_doa_labels']
        self._out_size = parameters['out_size']
        self.fc_out_size = parameters['out_size'] * parameters['num_doa_labels']

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_hidden_size),
            nn.BatchNorm1d(self.fc_hidden_size),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_hidden_size, self.fc_hidden_size),
            nn.BatchNorm1d(self.fc_hidden_size),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_hidden_size, self.fc_hidden_size),
            nn.BatchNorm1d(self.fc_hidden_size),
            nn.ReLU()
        )

        if self._out_size == 2 or self._out_size == 3:
            activate_func = nn.Tanh()   # For directly estimating (x, y, z) having -inf ~ +inf values
        else:
            activate_func = nn.Sigmoid()    # For estimating grids corresponding (x, y, z) on the unit sphere

        self.fc_final = nn.Sequential(
            nn.Linear(self.fc_hidden_size, self.fc_out_size),
            # nn.Sigmoid()
            activate_func
        )

    # def load_saved_tdoa_model(self, path):
    #     self._TDOA_model.load_state_dict(torch.load(path))
    def freeze_tdoa_model(self):
        for param in self._TDOA_model.parameters():
            param.requires_grad = False

    def unfreeze_tdoa_model(self):
        for param in self._TDOA_model.parameters():
            param.requires_grad = True

    def forward(self, tf):
        b, p_num, ch, h, w = tf.shape
        reshaped_tf = torch.reshape(tf, (-1, ch, h, w))
        # reconstruct_tf = torch.reshape(reshaped_tf, (b, p_num, ch, h, w))
        # tdoa_feature = self._TDOA_model(reshaped_tf)
        tdoa_feature, tdoa_out = self._TDOA_model(reshaped_tf)
        _, num_label, num_fr, num_delay = tdoa_out.shape
        tdoa_out = torch.reshape(tdoa_out, (b, p_num, num_label, num_fr, num_delay))

        ### Version 1
        # tdoa_feature = tdoa_feature.reshape(b, p_num, self._fr_len, self.input_features)
        # tdoa_feature = tdoa_feature.permute(0, 2, 3, 1)
        # tdoa_feature_out = tdoa_feature
        # tdoa_feature = tdoa_feature.reshape(b, self._fr_len, -1)
        # tdoa_feature = tdoa_feature.reshape(b*self._fr_len, -1)
        # MLP
        # fc_out = self.fc1(tdoa_feature)

        ### Version 2
        tdoa_out_feature = tdoa_out.permute(0, 3, 1, 2, 4)
        tdoa_feature_out = tdoa_out_feature
        tdoa_out_feature = tdoa_out_feature.reshape(b * self._fr_len, -1)
        # MLP
        fc_out = self.fc1(tdoa_out_feature)

        fc_out = self.fc2(fc_out)
        fc_out = self.fc3(fc_out)
        fc_out = self.fc_final(fc_out)

        if self._num_doa_labels == 1:
            # return fc_out.reshape(b, self._fr_len, -1)
            return fc_out.reshape(b, self._fr_len, -1), tdoa_out, tdoa_feature_out
        elif self._out_size == 2 or self._out_size == 3:
            return fc_out.reshape(b, self._fr_len, -1), tdoa_out, tdoa_feature_out
        else:
            return fc_out.reshape(b, self._fr_len, self._out_size, self._num_doa_labels), tdoa_out, tdoa_feature_out
            # return fc_out.reshape(b, self._fr_len, self._out_size, self._num_doa_labels), tdoa_out