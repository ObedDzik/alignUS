# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from DINO and timm library:
https://github.com/facebookresearch/dino
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
from typing import Callable, Literal, Sequence
from warnings import warn
import torch
import torch.nn as nn

from functools import partial
from .utils import trunc_normal_
from einops.layers.torch import Rearrange
from einops import pack, unpack
from .registry import register_model


def pair(obj):

    if isinstance(obj, Sequence):
        assert len(obj) == 2
        return obj
    else:
        return obj, obj


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values=0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class BasePatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self, x):
        """
        Turns image to patch tokens.

        Args:
            x: Tensor - B, C, H, W image

        Returns:
            Tensor - B, C, nPatchesH, nPatchesW patch embeddings.
        """


class PatchEmbed(BasePatchEmbed):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__(img_size, patch_size, in_chans)
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x)


class FFTPatchEmbed(BasePatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__(img_size, patch_size, in_chans)
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.fourier_patch_size = (patch_size // 2) + 1
        self.num_patches = num_patches

        self.proj = nn.Linear(
            in_chans * self.fourier_patch_size * self.patch_size, embed_dim
        )
        nn.init.normal_(self.proj.weight, 0, 0.0001)

    def get_patch_level_fft(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        PF = self.fourier_patch_size
        nP = H // P
        patches = x.resize(B, C, nP, P, nP, P)
        patches = patches.permute(0, 1, 2, 4, 3, 5)
        patches = torch.fft.rfft2(patches, norm="forward").abs().log()

        # sometimes patches has inf values, fill them
        patches[torch.isinf(patches)] = patches.max()

        return patches

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        PF = self.fourier_patch_size
        nP = H // P
        patches = self.get_patch_level_fft(x)
        patches_flat = patches.permute(0, 2, 3, 1, 4, 5).resize(B, nP, nP, C * P * PF)
        proj = self.proj(patches_flat).abs()
        return proj.permute(0, 3, 1, 2)


class FFTPatchEmbedV2(BasePatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__(img_size, patch_size, in_chans)

        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        freq_patch_size = patch_size
        freq_patch_height, freq_patch_width = pair(freq_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert (
            image_height % freq_patch_height == 0
            and image_width % freq_patch_width == 0
        ), "Image dimensions must be divisible by the freq patch size."

        patch_dim = in_chans * patch_height * patch_width
        freq_patch_dim = in_chans * 2 * freq_patch_height * freq_patch_width

        patch_height, patch_width = patch_size, patch_size
        h = image_height // patch_height
        w = image_width // patch_width
        patch_dims = image_height // patch_height * image_width // patch_width

        self.num_patches = h
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                p1=patch_height,
                p2=patch_width,
                c=in_chans,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
        )

        self.to_freq_patch = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> (b h w) c p1 p2",
                p1=freq_patch_height,
                p2=freq_patch_width,
            ),
        )

        self.to_freq_embedding = nn.Sequential(
            Rearrange(
                "(b d) c p1 p2 ri -> b d (p1 p2 ri c)",
                p1=freq_patch_height,
                p2=freq_patch_width,
                d=patch_dims,
            ),
            nn.LayerNorm(freq_patch_dim),
            nn.Linear(freq_patch_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
        )

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), Rearrange("b (h w) d -> b d h w", h=h, w=w)
        )

    def forward(self, x):
        patch_emb = self.to_patch_embedding(x)
        patch_for_freq = self.to_freq_patch(x)
        freqs = torch.fft.fft2(patch_for_freq)
        freqs = torch.view_as_real(freqs)
        freqs_emb = self.to_freq_embedding(freqs)

        merged_embs, _ = pack([freqs_emb, patch_emb], "b n *")

        out = self.to_out(merged_embs)
        return out


class ConvPatchEmbedWithNoOverlappingAcrossPatches(BasePatchEmbed):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_conv_layers=3,
        kernel_size=3,
    ):
        super().__init__(img_size, patch_size, in_chans)

        num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        nph = npw = img_size // patch_size
        self.split_into_patchs = Rearrange(
            "b c (nph ph) (npw pw) -> (b nph npw) c ph pw", nph=npw, npw=npw
        )
        self.conv_layers = torch.nn.ModuleList()
        in_chans = in_chans
        out_chans = 16
        for _ in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            in_chans = out_chans
            out_chans *= 2

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.proj_to_embedding_dim = nn.Conv2d(out_chans // 2, embed_dim, 1)
        self.to_patch_token_grid = Rearrange(
            "(b nph npw) c ph pw -> b c (nph ph) (npw pw)", nph=nph, npw=npw, pw=1, ph=1
        )

    def forward(self, x: torch.Tensor):
        x = self.split_into_patchs(x)

        for layer in self.conv_layers:
            x = layer(x)
            x = nn.MaxPool2d(2, 2)(x)
            x = x.relu()

        x = self.global_pool(x)
        x = self.proj_to_embedding_dim(x)
        x = self.to_patch_token_grid(x)

        return x


class FullConvPatchEmbedding(BasePatchEmbed):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        hidden_channels=[32, 128, 256],
    ):
        super().__init__(img_size, patch_size, in_chans)
        self.layers = []
        self.layers.append(
            nn.Conv2d(
                in_chans,
                hidden_channels[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        )
        self.layers.append(nn.GroupNorm(8, hidden_channels[0]))
        self.layers.append(nn.ReLU())

        for i in range(len(hidden_channels) - 1):
            self.layers.append(
                nn.Conv2d(
                    hidden_channels[i],
                    hidden_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.layers.append(nn.GroupNorm(8, hidden_channels[i + 1]))
            self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)

        cur_stride = 2 ** len(hidden_channels)
        # we have downsampled the image, but we need to downsample it further through the patch embedding.
        # total stride should be equal to patch_size
        patch_size_for_embed = patch_size // cur_stride

        self.to_embed = nn.Conv2d(
            hidden_channels[-1],
            embed_dim,
            kernel_size=patch_size_for_embed,
            stride=patch_size_for_embed,
            padding=0,
            bias=False,
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = self.to_embed(x)
        return x


patch_embed_layers_classes = {
    "patch_embed": PatchEmbed,
    "fft_patch_embed": FFTPatchEmbed,
    "fft_patch_embed_v2": FFTPatchEmbedV2,
    "conv_patch_embed": partial(
        ConvPatchEmbedWithNoOverlappingAcrossPatches, num_conv_layers=3, kernel_size=3
    ),
    "full_conv_patch_embed": FullConvPatchEmbedding,
}


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0,
        use_mean_pooling=False,
        masked_im_modeling=False,
        patch_embed_factory: Callable | str = "patch_embed",
        patch_embed_kw: dict = {},
        n_cls_tokens=1,
        use_sincos_pos_embed=False,
        freeze_pos_embed=False,
        return_all_tokens=False, 
        return_feature_map=False,
        pool_mode: Literal['cls', 'avg_patch', 'reg'] | str = 'cls',
        output_format=None,
    ):
        super().__init__()

        if not isinstance(img_size, Sequence):
            img_size = [img_size]

        self.return_all_tokens = return_all_tokens
        self.return_feature_map = return_feature_map
        self.num_features = self.embed_dim = embed_dim
        self.n_cls_tokens = n_cls_tokens
        self.token_pool_mode = pool_mode
        self.output_format = output_format

        if isinstance(patch_embed_factory, str):
            patch_embed_factory = patch_embed_layers_classes[patch_embed_factory]

        self.patch_embed = patch_embed_factory(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            **patch_embed_kw,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_cls_tokens + num_patches, embed_dim)
        )
        if use_sincos_pos_embed:
            _sincos_pos_embed = posemb_sincos_2d(
                int(num_patches**0.5), int(num_patches**0.5), embed_dim
            ).unsqueeze(0)
            self.pos_embed.data[:, n_cls_tokens:, :] = _sincos_pos_embed
        if freeze_pos_embed:
            self.pos_embed.requires_grad = False

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))
        else:
            self.masked_embed = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, : self.n_cls_tokens]
        patch_pos_embed = self.pos_embed[:, self.n_cls_tokens :]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None, additional_tokens=None):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            if self.masked_im_modeling:
                x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
            else:
                warn(
                    f"Received mask but model is not configured for masked image modeling!"
                )

        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        cls_tokens = x[:, : self.n_cls_tokens, :]
        patch_tokens = x[:, self.n_cls_tokens :, :]
        if additional_tokens is not None:
            assert additional_tokens.shape[0] == B
            assert additional_tokens.shape[-1] == cls_tokens.shape[-1]
            cls_tokens = torch.cat((cls_tokens, additional_tokens), dim=1)
        x = torch.cat((cls_tokens, patch_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, return_all_tokens=False, mask=None, additional_tokens=None, return_feature_map=False):

        outputs_dict = {}

        # mim
        x = self.prepare_tokens(x, mask=mask, additional_tokens=additional_tokens)

        layer_outputs = []
        for blk in self.blocks:
            x = blk(x)
            layer_outputs.append(x)
        outputs_dict["layer_outputs"] = layer_outputs

        feature_maps = []
        for i, layer_output in enumerate(layer_outputs):
            feature_map = layer_output[:, self.n_cls_tokens :, :]
            B, N, C = feature_map.shape
            H = W = int(math.sqrt(N))
            feature_map = feature_map.reshape(B, H, W, C).permute(0, 3, 1, 2)
            feature_maps.append(feature_map)
        outputs_dict["feature_maps"] = feature_maps

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))
        
        outputs_dict["last_layer_output"] = x 
        outputs_dict["cls_tokens"] = x[:, : self.n_cls_tokens, :]
        outputs_dict["patch_tokens"] = x[:, self.n_cls_tokens :, :]
        outputs_dict["all_tokens"] = x

        if return_all_tokens or self.return_all_tokens:
            return x

        feature_map = x[:, self.n_cls_tokens:, :]
        B, N, C = feature_map.shape
        H = W = int(math.sqrt(N))
        feature_map = feature_map.reshape(B, H, W, C).permute(0, 3, 1, 2)
        outputs_dict["last_feature_map"] = outputs_dict['feature_map'] = feature_map

        if return_feature_map or self.return_feature_map:
            return feature_map
        
        if self.token_pool_mode == 'cls': 
            pre_logits = x[:, 0]
        elif self.token_pool_mode.startswith('reg'): 
            reg_idx = int(self.token_pool_mode.split('_')[-1])
            pre_logits = x[:, reg_idx + 1]
        elif self.token_pool_mode == 'avg_patch': 
            pre_logits = x[:, self.n_cls_tokens:, :].mean(dim=1)
        else: 
            raise ValueError(f"Unknown pool mode {self.token_pool_mode}")

        outputs_dict["pre_logits_clstoken"] = outputs_dict['image_level_feature'] = pre_logits
        head_output = self.head(pre_logits)
        outputs_dict["head_output"] = head_output

        if self.output_format is None: 
            return head_output
        else: 
            return outputs_dict

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_num_layers(self):
        return len(self.blocks)

    def get_class_token(self, x) -> torch.Tensor:
        tokens = self(x, return_all_tokens=False)
        return tokens

    def get_feature_map(self, x) -> torch.Tensor:
        feature_map = self(x, return_feature_map=True)
        return feature_map


class VisionTransformerWithDenseLinearProjection(nn.Module):
    def __init__(
        self,
        vit_model: VisionTransformer,
        num_classes: int,
        mode: Literal["cls", "all", "feature_map"] = "cls",
        freeze_backbone=False,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.num_classes = num_classes
        self.head = nn.Linear(vit_model.embed_dim, num_classes)
        self.mode = mode
        self.freeze_backbone = freeze_backbone

    def forward(self, x):
        if self.mode == "cls":
            with torch.set_grad_enabled(not self.freeze_backbone):
                x = self.vit_model(x)  # B, C
            return self.head(x)

        elif self.mode == "all":
            with torch.set_grad_enabled(not self.freeze_backbone):
                x = self.vit_model(x, return_all_tokens=True)
            # B, N, C
            return self.head(x)

        elif self.mode == "feature_map":
            with torch.set_grad_enabled(not self.freeze_backbone):
                x = self.vit_model.get_feature_map(x)  # B, C, H, W
            proj = self.head(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return proj

        else:
            raise ValueError(f"Invalid mode: {self.mode}")


def _apply_lora_to_vit(module, r=8):
    import loralib
    for name, submodule in module.named_children(): 
        if name.endswith('qkv'):
            new_submodule = loralib.MergedLinear(
                submodule.in_features,
                submodule.out_features,
                r=r,
                enable_lora=[True, False, False]
            )
            new_submodule.load_state_dict(submodule.state_dict(), strict=False)
        else: 
            new_submodule = _apply_lora_to_vit(submodule, r=r)
        setattr(module, name, new_submodule)

    loralib.mark_only_lora_as_trainable(module)
    return module


def _create_vit(lora_r=None, **kwargs):
    model = VisionTransformer(**kwargs)
    if lora_r is not None: 
        model = _apply_lora_to_vit(model, r=lora_r)

    return model


@register_model
def vit_tiny(patch_size=16, **kwargs):
    model_args = dict(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return _create_vit(**dict(model_args, **kwargs))


@register_model
def vit_small(patch_size=16, **kwargs):
    model_args = dict(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return _create_vit(**dict(model_args, **kwargs))


@register_model
def vit_base(patch_size=16, **kwargs):
    model_args = dict(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return _create_vit(**dict(model_args, **kwargs))


@register_model
def vit_large(patch_size=16, **kwargs):
    model_args = dict(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return _create_vit(**dict(model_args, **kwargs))


@register_model
def deit_vit_tiny(**kwargs):
    model_args = dict(
        n_cls_tokens=2,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return _create_vit(**dict(model_args, **kwargs))


@register_model
def deit_vit_small(**kwargs):
    model_args = dict(
        n_cls_tokens=2,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return _create_vit(**dict(model_args, **kwargs))


@register_model
def deit_vit_base(**kwargs):
    return vit_base(n_cls_tokens=2, **kwargs)


@register_model
def deit_vit_large(**kwargs):
    return vit_large(n_cls_tokens=2, **kwargs)


def vit_small_fft_v2(patch_size=16, **kwargs):
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbedV2, **kwargs)


def vit_small_fft_v1(patch_size=16, **kwargs):
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbed, **kwargs)


if __name__ == "__main__":
    image = torch.randn(1, 3, 224, 224)
    patch_embed = ConvPatchEmbedWithNoOverlappingAcrossPatches(
        224,
        16,
        3,
    )
    patch_embed_2 = PatchEmbed(224, 16, 3)

    print(
        VisionTransformer(patch_embed_factory="conv_patch_embed")
        .forward(image, return_all_tokens=True)
        .shape
    )
