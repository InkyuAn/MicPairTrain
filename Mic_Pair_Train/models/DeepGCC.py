import torch
from torch import nn


class deep_gcc(nn.Module):
    def __init__(self, *, kernel_size=5):
        super().__init__()

        en_input_ch = [1, 2, 8, 32]
        en_output_ch = [2, 8, 32, 128]
        de_input_ch = [128, 32, 8, 2]
        de_output_ch = [32, 8, 2, 1]

        # self._input_length = length
        self._encoder = nn.ModuleList()
        for input_ch, output_ch in zip(en_input_ch, en_output_ch):
            tmp_en_block = nn.Sequential(
                nn.Conv1d(input_ch, output_ch, kernel_size=kernel_size, padding=2),
                nn.MaxPool1d(2, stride=2),
                nn.BatchNorm1d(output_ch),
                nn.ReLU()
            )
            # tmp_en_block = nn.ModuleList()
            # tmp_en_block.append(nn.Conv1d(input_ch, output_ch, kernel_size=kernel_size, padding=2))
            # tmp_en_block.append(nn.MaxPool1d(2, stride=2))
            # tmp_en_block.append(nn.BatchNorm1d(output_ch))
            # tmp_en_block.append(nn.ReLU())
            self._encoder.append(tmp_en_block)

        self._decoder = nn.ModuleList()
        for input_ch, output_ch in zip(de_input_ch, de_output_ch):
            tmp_de_block = nn.Sequential(
                nn.Conv1d(input_ch, output_ch, kernel_size=kernel_size, padding=2),
                nn.Upsample(scale_factor=2),
                nn.BatchNorm1d(output_ch),
                nn.ReLU()
            )

            # tmp_de_block = nn.ModuleList()
            # tmp_de_block.append(nn.Conv1d(input_ch, output_ch, kernel_size=kernel_size, padding=2),)
            # tmp_de_block.append(nn.Upsample(scale_factor=2),)
            # tmp_de_block.append(nn.BatchNorm1d(output_ch))
            # tmp_de_block.append(nn.ReLU())

            self._decoder.append(tmp_de_block)

    def forward(self, x):
        nb, nf, np, ndelay = x.shape

        x = torch.reshape(x, (-1, 1, ndelay))

        for en_block in self._encoder:
            x = en_block(x)
            # print("Start")
            #
            # for i, en_block_elem in enumerate(en_block):
            #     x = en_block_elem(x)
            #     print("Idx", i)
            #
            # print("End")
        # print("End Encoder")
        for de_block in self._decoder:
            x = de_block(x)
            # print("1")
            # for i, de_block_elem in enumerate(de_block):
            #     x = de_block_elem(x)
            #     print("Idx", i)
            # print("End")
        x = torch.reshape(x, (nb, nf, np, ndelay))
        return x