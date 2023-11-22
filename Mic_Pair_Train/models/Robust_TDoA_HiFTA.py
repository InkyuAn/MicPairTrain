import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# from .modeling_resnet import ResNetV2
from .Learnable_mel_filter import learnable_mel_scale_filter

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(val, divisor):
    return (val % divisor) == 0

def unfold_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads =  heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

# class TNT_stft(nn.Module):
class RobustTDoA_HiFTA(nn.Module):
    def __init__(
        self,
        *,
        stft_size,
        stft_ch,
        patch_w_num,
        patch_h_num=1,
        pixel_w_num=1,
        pixel_h_num,
        patch_dim,
        pixel_dim,
        # patch_size,
        # pixel_size,
        # patch_num,
        # pixel_num,
        depth,
        delay_len,
        heads = 8,
        dim_head = 512,
        mlp_head_dim = 128,
        ff_dropout = 0.,
        attn_dropout = 0.,
        w_hifta = True,
        num_labels = 1,
        use_hybrid_input=False,
        return_features=False,
        use_mel_filter=False,
        fs=24000,
        unfold_args = None
    ):
        super().__init__()

        self._w_hifta = w_hifta
        self._return_feat = return_features

        num_pixels = pixel_h_num * pixel_w_num

        stft_h_size, stft_w_size = stft_size

        self._use_mel_filter = use_mel_filter
        if self._use_mel_filter:
            self._learnable_mel_filter = learnable_mel_scale_filter(n_freq=stft_h_size,
                                                                    n_ch=stft_ch,
                                                                    fs=fs,
                                                                    n_mels=64,
                                                                    n_filter=16)
            stft_h_size = 64
            stft_ch = 16
        ### Hybrid
        self._use_hybrid_input = use_hybrid_input
        self._hybrid_model = nn.ModuleList()
        if self._use_hybrid_input:
            # self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1, input_ch=stft_ch)
            cnn_filter_num = 128
            cnn_kernel_size = (3, 3)
            cnn_t_pool_size = (1, 1, 1)
            cnn_f_pool_size = (8, 8, 4)
            pads = ((1, 1), (1, 1), (1, 1))
            # pads = ((0, 0), (0, 0), (0, 0))

            self._hybrid_model.append(
                nn.Sequential(
                    nn.Conv2d(4, cnn_filter_num, cnn_kernel_size, padding=pads[0]),
                    nn.BatchNorm2d(cnn_filter_num),
                    nn.ReLU(),
                    nn.MaxPool2d((cnn_f_pool_size[0], cnn_t_pool_size[0]))
                )
            )
            self._hybrid_model.append(
                nn.Sequential(
                    nn.Conv2d(cnn_filter_num, cnn_filter_num*2, cnn_kernel_size, padding=pads[1]),
                    nn.BatchNorm2d(cnn_filter_num*2),
                    nn.ReLU(),
                    nn.MaxPool2d((cnn_f_pool_size[1], cnn_t_pool_size[1]))
                )
            )
            self._hybrid_model.append(
                nn.Sequential(
                    # nn.Conv2d(cnn_filter_num, cnn_filter_num, cnn_kernel_size, padding=(1, 0)),
                    nn.Conv2d(cnn_filter_num*2, cnn_filter_num*4, cnn_kernel_size, padding=pads[2]),
                    nn.BatchNorm2d(cnn_filter_num*4),
                    nn.ReLU(),
                    nn.MaxPool2d((cnn_f_pool_size[2], cnn_t_pool_size[2])),
                    # Rearrange('b c h w -> b (c h) w', c=cnn_filter_num*4, h=1, w=stft_w_size)
                    Rearrange('b c h w -> b h c w', c=cnn_filter_num * 4, h=1, w=stft_w_size)
                )
            )
            stft_ch = 1
            stft_h_size = cnn_filter_num*4

        patch_h_size = stft_h_size
        patch_w_size = stft_w_size // patch_w_num
        pixel_h_size = stft_h_size // pixel_h_num
        pixel_w_size = patch_w_size

        self.stft_size = stft_size
        self.patch_size = (patch_h_size, patch_w_size)

        # assert divisible_by(stft_w_size, patch_w_size), 'image size must be divisible by patch size'
        # assert divisible_by(stft_h_size, patch_h_size), 'image size must be divisible by patch size'
        # assert divisible_by(patch_w_size, pixel_w_size), 'patch size must be divisible by pixel size for now'
        # assert divisible_by(patch_h_size, pixel_h_size), 'patch size must be divisible by pixel size for now'

        # num_patch_tokens = (image_size // patch_size) ** 2
        num_patch_tokens = patch_h_num * patch_w_num
        self._patch_h_num = patch_h_num
        self._patch_w_num = patch_w_num

        self.patch_tokens = nn.Parameter(torch.randn(num_patch_tokens, patch_dim))

        # unfold_args = default(unfold_args, (pixel_size, pixel_size, 0))
        # unfold_args = (*unfold_args, 0) if len(unfold_args) == 2 else unfold_args
        # kernel_size, stride, padding = unfold_args

        kernel_patch_size = (patch_h_size, patch_w_size)
        stride_patch = (patch_h_size, patch_w_size)
        padding_patch = (0, 0)

        kernel_pixel_size = (pixel_h_size, pixel_w_size)
        stride_pixel = (pixel_h_size, pixel_w_size)
        padding_pixel = (0, 0)

        # pixel_width = unfold_output_size(patch_size, kernel_size, stride, padding)
        # num_pixels = pixel_width ** 2

        # self.to_pixel_tokens = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size),
        #     nn.Unfold(kernel_size = kernel_size, stride = stride, padding = padding),
        #     Rearrange('... c n -> ... n c'),
        #     nn.Linear(3 * kernel_size ** 2, pixel_dim)
        # )

        self.to_pixel_tokens = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size),
            nn.Unfold(kernel_size=kernel_patch_size, stride=stride_patch, padding=padding_patch),
            Rearrange('b (c ph pw) (ph_n pw_n) -> (b ph_n pw_n) c ph pw', c=stft_ch, ph=patch_h_size, pw=patch_w_size,
                      ph_n=patch_h_num, pw_n=patch_w_num),
            nn.Unfold(kernel_size=kernel_pixel_size, stride=stride_pixel, padding=padding_pixel),
            Rearrange('... c n -> ... n c'),
            nn.Linear(stft_ch * pixel_h_size * pixel_w_size, pixel_dim)
        )


        self.patch_pos_emb = nn.Parameter(torch.randn(num_patch_tokens, patch_dim))
        self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))

        layers = nn.ModuleList([])
        for _ in range(depth):

            pixel_to_patch = nn.Sequential(
                nn.LayerNorm(pixel_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(pixel_dim * num_pixels, patch_dim),
            )

            layers.append(nn.ModuleList([
                PreNorm(pixel_dim, Attention(dim = pixel_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(pixel_dim, FeedForward(dim = pixel_dim, dropout = ff_dropout)),
                pixel_to_patch,
                PreNorm(patch_dim, Attention(dim = patch_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(patch_dim, FeedForward(dim = patch_dim, dropout = ff_dropout)),
            ]))

        self.layers = layers

        self._mlp_head = nn.Sequential(
                nn.LayerNorm(patch_dim),
                Rearrange('b p d -> (b p) d', p=patch_h_num*patch_w_num, d=patch_dim),
                nn.Linear(patch_dim, mlp_head_dim),
            )

        ### For multiple-labels (previous version: binary class)
        if num_labels > 1:
            self.mlp_out = nn.Sequential(
                nn.Linear(mlp_head_dim, delay_len * num_labels),
                Rearrange('(b p) (d c) -> b c p d', p=patch_h_num * patch_w_num, d=delay_len, c=num_labels),
                nn.Sigmoid()
            )
        else:   # Binary classification
            self.mlp_out = nn.Sequential(
                nn.Linear(mlp_head_dim, delay_len),
                Rearrange('(b p) d -> b p d', p=patch_h_num * patch_w_num, d=delay_len),
                nn.Sigmoid()
            )

        # if num_labels > 1:
        #     self.mlp_head = nn.Sequential(
        #         nn.LayerNorm(patch_dim),
        #         Rearrange('b p d -> (b p) d', p=patch_h_num*patch_w_num, d=patch_dim),
        #         nn.Linear(patch_dim, delay_len * num_labels),
        #         Rearrange('(b p) (d c) -> b c p d', p=patch_h_num * patch_w_num, d=delay_len, c=num_labels),
        #         nn.Sigmoid()
        #     )
        # else:   # Binary classification
        #     self.mlp_head = nn.Sequential(
        #         nn.LayerNorm(patch_dim),
        #         Rearrange('b p d -> (b p) d', p=patch_h_num * patch_w_num, d=patch_dim),
        #         nn.Linear(patch_dim, delay_len),
        #         Rearrange('(b p) d -> b p d', p=patch_h_num * patch_w_num, d=delay_len),
        #         nn.Sigmoid()
        #     )

    def forward(self, x):
        b, _, stft_h_size, stft_w_size = x.shape
        patch_h_size, patch_w_size = self.patch_size
        # b, _, h, w, patch_size, image_size = *x.shape, self.patch_size, self.image_size
        # assert divisible_by(h, patch_size) and divisible_by(w, patch_size), f'height {h} and width {w} of input must be divisible by the patch size'

        # num_patches_h = stft_h_size // patch_h_size
        # num_patches_w = (stft_w_size) // (patch_w_size)
        num_patches_h = self._patch_h_num
        num_patches_w = self._patch_w_num
        n = num_patches_w * num_patches_h

        ### Hybrid model
        for hybrid_model in self._hybrid_model:
            x = hybrid_model(x)

        ### Mel
        if self._use_mel_filter:
            x = self._learnable_mel_filter(x)

        # pixels = self.to_pixel_tokens(x)
        pixels = self.to_pixel_tokens(x)
        # pixels = self.to_pixel_tokens_af(pixels)
        # Need to be checked, 20220614 IK
        # patches = repeat(self.patch_tokens[:n], 'n d -> b n d', b = b)
        # patches += rearrange(self.patch_pos_emb[:n], 'n d -> () n d')
        patches = repeat(self.patch_tokens[:n], 'n d -> b n d', b=b) + rearrange(self.patch_pos_emb[:n],
                                                                                 'n d -> () n d')

        pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')

        ### For debugging
        if self._w_hifta:
            for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:

                pixels = pixel_attn(pixels) + pixels
                pixels = pixel_ff(pixels) + pixels

                patches_residual = pixel_to_patch_residual(pixels)

                patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches_h, w = num_patches_w)
                # patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
                patches = patches + patches_residual

                patches = patch_attn(patches) + patches
                patches = patch_ff(patches) + patches

            # cls_token = patches[:, 0]
        else:   # Without HiFTA
            for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:
                patches_residual = pixel_to_patch_residual(pixels)

                patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h=num_patches_h,
                                             w=num_patches_w)
                # patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
                patches = patches + patches_residual

        y_feat = self._mlp_head(patches)

        y_out = self.mlp_out(y_feat)

        if self._return_feat:
            return y_feat, y_out
        else:
            return y_out
        # if self._return_feat:
        #     return y
        # return self.mlp_out(y)

    # self._return_feat
