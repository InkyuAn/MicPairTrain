import os
import sys

import torch
import torch.nn as nn
import math

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

#from Est_TDOA.models.Transformer import Transformer
from .Transformer import Transformer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .Learnable_mel_filter import learnable_mel_scale_filter

def center_trim(tensor, reference):
    """
    Trim a tensor to match with the dimension of `reference`.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2:-(diff - diff // 2)]
    return tensor


'''
##########################################################################################
################# Input: STFT ############################################################
##########################################################################################
'''
# class ShiftPairNet_STFT_Transformer(nn.Module):
class RobustTDoA_Trans(nn.Module):
    def __init__(self, *, stft_size, stft_ch,
                 patch_num, # patch_dim,
                 delay_len,
                 depth=1,
                 heads=16,
                 dim_head=64,
                 embed_dim=256,
                 mlp_dim=256,
                 mlp_head_dim=128,
                 # pool='cls',
                 dropout=0.,
                 emb_dropout=0.,
                 num_label=1,
                 return_features=False,
                 use_mel_filter=False,
                 fs=24000):
        super().__init__()

        self._return_feat = return_features

        stft_h_size, stft_w_size = stft_size
        patch_h_num, patch_w_num = patch_num
        self._use_mel_filter = use_mel_filter
        if self._use_mel_filter:
            self._learnable_mel_filter = learnable_mel_scale_filter(n_freq=stft_h_size,
                                                                    n_ch=stft_ch,
                                                                    fs=fs,
                                                                    n_mels=64,
                                                                    n_filter=16)
            stft_h_size = 64
            stft_ch = 16

        patch_h_size = stft_h_size // patch_h_num
        patch_w_size = stft_w_size // patch_w_num

        self.num_patches = patch_h_num * patch_w_num
        self.embed_dim = embed_dim
        # patch_dim_1d = patch_dim[0] * patch_dim[1] * stft_ch
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Input shape is [ Batch, (num_delay+1), audio_length ]
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h phd) (w pwd) -> (b h w) (c phd pwd)', c=stft_ch, phd=patch_h_size, pwd=patch_w_size),
            nn.Linear(stft_ch*patch_h_size*patch_w_size, embed_dim),
            Rearrange('(b h w) ed -> b (h w) ed', h=patch_h_num, w=patch_w_num, ed=embed_dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(self.num_patches, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.pool = pool
        self.to_latent = nn.Identity()

        # num_classes = 2
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, num_classes)
        # )

        self._mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Rearrange('b (h w) ed -> (b w) (h ed)', h=patch_h_num, w=patch_w_num, ed=embed_dim),
            nn.Linear(embed_dim * patch_h_num, mlp_head_dim),
        )

        if num_label > 1:
            # out_classes = delay_len * num_label
            self._mlp_out = nn.Sequential(
                # nn.LayerNorm(embed_dim),
                # Rearrange('b (h w) ed -> (b w) (h ed)', h=patch_h_num, w=patch_w_num, ed=embed_dim),
                # nn.Linear(embed_dim * patch_h_num, delay_len * num_label),
                nn.Linear(mlp_head_dim, delay_len * num_label),
                Rearrange('(b w) (d nl) -> b nl w d', w=patch_w_num, d=delay_len, nl=num_label),
                nn.Sigmoid()
            )
        else:
            self._mlp_out = nn.Sequential(
                # nn.LayerNorm(embed_dim),
                # Rearrange('b (h w) ed -> (b w) (h ed)', h=patch_h_num, w=patch_w_num, ed=embed_dim),
                # nn.Linear(embed_dim * patch_h_num, delay_len),
                nn.Linear(mlp_head_dim, delay_len),
                Rearrange('(b w) d -> b w d', w=patch_w_num, d=delay_len),
                nn.Sigmoid()
            )


        # self.mlp_head_bf = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     Rearrange('b (h w) ed -> (b w) (h ed)', h=patch_num[0], w=patch_num[1], ed=embed_dim),
        # )
        # self.mlp_head_af = nn.Sequential(
        #     nn.Linear(embed_dim*patch_num[0], out_classes),
        #     Rearrange('(b w) oc -> b w oc', w=patch_num[1], oc=out_classes),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        b, _, stft_h_size, stft_w_size = input.shape

        ### Mel
        if self._use_mel_filter:
            input = self._learnable_mel_filter(input)

        x = self.to_patch_embedding(input)

        # x += self.pos_embedding
        x += repeat(self.pos_embedding, 'np ed -> b np ed', b=b)
        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        # x = self.mlp_head_bf(x)
        # x = self.mlp_head_af(x)
        x_feat = self._mlp_head(x)

        x_out = self._mlp_out(x_feat)

        if self._return_feat:
            return x_feat, x_out
        else:
            return x_out

        # if self._return_feat:
        #     return x
        # else:
        #     return self._mlp_out(x)



