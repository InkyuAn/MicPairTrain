import torch
import torch.fft
import torch.nn as nn


from .DeepGCC import deep_gcc

class DeepGCC_DOA(nn.Module):
    def __init__(self, num_pair, out_size=360, num_doa_labels=1):
        super(DeepGCC_DOA, self).__init__()

        self._eps = torch.finfo(torch.float).eps

        self._len_fr = 10
        self._ngcc_per_fr = 5
        self._npair = num_pair
        self._len_gcc = 128

        self.fc_input_size = self._ngcc_per_fr * self._npair * self._len_gcc
        self.fc_hidden_size = 1000

        self._num_doa_labels = num_doa_labels
        self._out_size = out_size
        self.fc_out_size = out_size * num_doa_labels

        ### Design Network Architecture
        self._TDOA_model = deep_gcc()

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
            activate_func
        )

    def freeze_tdoa_model(self):
        for param in self._TDOA_model.parameters():
            param.requires_grad = False

    def unfreeze_tdoa_model(self):
        for param in self._TDOA_model.parameters():
            param.requires_grad = True

    def forward(self, gcc):

        nb, nf, np, ndelay = gcc.shape

        gccout = self._TDOA_model(gcc)

        _, ntbin, _, len_gcc = gccout.shape
        assert (np == self._npair) and (len_gcc == self._len_gcc), "The shapes of Model input and prediction are not same ..."

        rs_gccout = torch.reshape(gccout, (nb, self._len_fr, self._ngcc_per_fr, self._npair, self._len_gcc))
        rs_gccout = torch.reshape(rs_gccout, (nb * self._len_fr, self._ngcc_per_fr * self._npair * self._len_gcc))

        # MLP
        fc_out = self.fc1(rs_gccout)
        fc_out = self.fc2(fc_out)
        fc_out = self.fc3(fc_out)
        fc_out = self.fc_final(fc_out)

        if self._num_doa_labels == 1:
            return fc_out.reshape(nb, self._len_fr, -1), gccout
        elif self._out_size == 2 or self._out_size == 3:
            return fc_out.reshape(nb, self._len_fr, -1), gccout
        else:
            return fc_out.reshape(nb, self._len_fr, self._out_size, self._num_doa_labels), gccout