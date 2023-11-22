import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .Learnable_mel_filter import learnable_mel_scale_filter

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class CRNNpair(nn.Module):
    def __init__(self, fr_len=10, cnn_filter_num=128, cnn_kernel_size=(3, 3), cnn_t_pool_size=(2, 2, 1), cnn_f_pool_size=(8, 8, 4),
                 pads=((1, 1), (1, 1), (1, 0)),
                 rnn_size=128, fc_size=128, out_size=51, input_ch=4, stft_h_size=960,
                 num_labels=1, return_features=False,
                 use_middle_fc=True,
                 use_mel_filter=False,
                 fs=24000):
        super().__init__()
        self.fr_len = fr_len
        self._return_feat = return_features
        self._use_middle_fc = use_middle_fc

        self._use_mel_filter = use_mel_filter
        if self._use_mel_filter:
            self._learnable_mel_filter = learnable_mel_scale_filter(n_freq=stft_h_size,
                                                                    n_ch=input_ch,
                                                                    # fs=24000,
                                                                    fs=fs,
                                                                    n_mels=64,
                                                                    n_filter=16)
            # stft_h_size = 64
            input_ch = 16
            # self.fc_middle = nn.Sequential(
            #     TimeDistributed(nn.Linear(rnn_size, rnn_size), batch_first=True),
            # )
        # else:
        #     self.fc_middle = nn.Sequential(
        #         TimeDistributed(nn.Linear(1920, rnn_size), batch_first=True),
        #     )

        if self._use_middle_fc:
            self.fc_middle = nn.Sequential(
                        TimeDistributed(nn.Linear(1920, rnn_size), batch_first=True),
                    )

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_ch, cnn_filter_num, cnn_kernel_size, padding=pads[0]),
            nn.BatchNorm2d(cnn_filter_num),
            nn.ReLU(),
            nn.MaxPool2d((cnn_f_pool_size[0], cnn_t_pool_size[0]))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_filter_num, cnn_filter_num, cnn_kernel_size, padding=pads[1]),
            nn.BatchNorm2d(cnn_filter_num),
            nn.ReLU(),
            nn.MaxPool2d((cnn_f_pool_size[1], cnn_t_pool_size[1]))
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(cnn_filter_num, cnn_filter_num, cnn_kernel_size, padding=(1, 0)),
            nn.Conv2d(cnn_filter_num, cnn_filter_num, cnn_kernel_size, padding=pads[2]),
            nn.BatchNorm2d(cnn_filter_num),
            nn.ReLU(),
            nn.MaxPool2d((cnn_f_pool_size[2], cnn_t_pool_size[2]))
        )
        self.conv_filter_num = cnn_filter_num

        self._hidden_gru_dim = rnn_size
        self.rnn1 = nn.GRU(rnn_size, rnn_size, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(rnn_size, rnn_size, batch_first=True, bidirectional=True)

        # self.fc1 = nn.Sequential(
        #     TimeDistributed(nn.Linear(fc_size, fc_size), batch_first=True),
        # )
        self.fc1 = nn.Sequential(
            TimeDistributed(nn.Linear(rnn_size, fc_size), batch_first=True),
        )

        # self.fc2 = nn.Sequential(
        #     TimeDistributed(nn.Linear(fc_size, out_size), batch_first=True),
        #     nn.Sigmoid()
        # )

        if num_labels > 1:
            self.fc2 = nn.Sequential(
                TimeDistributed(nn.Linear(fc_size, out_size*num_labels), batch_first=True),
                Rearrange('b f (d c) -> b c f d', f=fr_len, d=out_size, c=num_labels),
                nn.Sigmoid()
            )
        else:   # Binary classification
            self.fc2 = nn.Sequential(
                TimeDistributed(nn.Linear(fc_size, out_size), batch_first=True),
                nn.Sigmoid()
            )


    def forward(self, x):
        # x: [Batch, Ch, Frame, Freq.]
        B, _, _, _ = x.shape

        ### Mel
        if self._use_mel_filter:
            x = self._learnable_mel_filter(x)

        conv_out = self.conv1(x)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)

        conv_out = conv_out.permute(0, 3, 1, 2)
        # conv_out = conv_out.view(-1, self.fr_len, self.conv_filter_num)
        # conv_out = conv_out.view(B, self.fr_len, -1)
        conv_out = conv_out.reshape(B, self.fr_len, -1)

        if self._use_middle_fc:
            conv_out = self.fc_middle(conv_out)

        # print("Debugging")
        # conv_out = conv_out.permute(0, 2, 1, 3)  # [batch, 64, 10, 2] --> [batch, 10, 64, 2]
        # conv_out_shape = conv_out.shape
        # conv_out = conv_out.reshape(conv_out_shape[0], conv_out_shape[1], -1)

        self.rnn1.flatten_parameters()
        gru_out, next_hidden = self.rnn1(conv_out)
        gru_out = (gru_out[:, :, :self._hidden_gru_dim] * gru_out[:, :, self._hidden_gru_dim:])
        # gru_out = self.rnn1_dropout(gru_out)
        self.rnn2.flatten_parameters()
        gru_out, next_hidden = self.rnn2(gru_out)
        gru_out = (gru_out[:, :, :self._hidden_gru_dim] * gru_out[:, :, self._hidden_gru_dim:])
        # gru_out = self.rnn2_dropout(gru_out)

        tdoa_feat = self.fc1(gru_out)
        tdoa_out = self.fc2(tdoa_feat)

        if self._return_feat:  # Return 128 dim features
            return tdoa_feat, tdoa_out
        else:

            return tdoa_out

        # fc_out = self.fc1(gru_out)
        # if self._return_feat:   # Return 128 dim features
        #     return fc_out
        # fc_out = self.fc2(fc_out)
        # return fc_out