from torch import nn
import torch
from external_libs.medsam.segment_anything.build_sam import sam_model_registry
from external_libs.medsam.segment_anything.modeling.image_encoder import (
    ImageEncoderViT,
    Block,
    Attention,
    add_decomposed_rel_pos,
    window_partition,
    window_unpartition,
)
from external_libs.medsam.segment_anything.modeling import Sam, image_encoder
import os
import math
import einops


def build_medsam():
    """
    Builds the MedSAM model by building the SAM model and loading the medsam checkpoint.
    """
    MEDSAM_CHECKPOINT = os.environ.get("MEDSAM_CHECKPOINT")
    if MEDSAM_CHECKPOINT is None:
        raise ValueError(
            "MEDSAM_CHECKPOINT environment variable must be set to use medsam."
        )
    checkpoint = MEDSAM_CHECKPOINT
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    return model


class MedSAMIBot(nn.Module):
    def __init__(
        self,
        freeze_pos_embed=False,
        version="1",
        n_cls_tokens=1,
        masked_im_modeling=None, # api compatibility with ibot script
        medsam_version='medsam',
        delete_neck=True,
    ):
        # keep defaults for now
        super().__init__()

        self.n_cls_tokens = n_cls_tokens
        
        if medsam_version == 'medsam': 
            medsam_model = build_medsam()
        elif medsam_version == 'medsam_adapter':
            from medAI.modeling.sam import build_adapter_medsam_256
            medsam_model = build_adapter_medsam_256()
        else: 
            raise ValueError(f'{medsam_model}?')

        image_encoder = medsam_model.image_encoder

        if delete_neck: 
            del (
                image_encoder.neck
            )  # parameters of neck are unused, delete it to avoid DDP "unused parameter" errors

        if version == "1":
            if n_cls_tokens > 1:
                raise ValueError(f"Version 1 is only compatible with 1 class token")
            self.image_encoder_wrapped = ImageEncoderViTWithClassTokenAndMasking(
                medsam_model.image_encoder
            )
        elif version == "2":
            self.image_encoder_wrapped = ImageEncoderViTWithClassTokenAndMaskingV2(
                medsam_model.image_encoder,
                n_class_tokens=n_cls_tokens,
            )

        if freeze_pos_embed:
            for p in self.image_encoder_wrapped.image_encoder.patch_embed.parameters():
                p.requires_grad = False

    def forward(self, image, mask=None, return_all_tokens=True):
        outputs = self.image_encoder_wrapped(image, mask)
        return outputs

    def get_feature_map(self, x):
        outputs = self.image_encoder_wrapped(x)
        outputs = outputs[:, self.n_cls_tokens :, :]
        B = outputs.shape[0]
        H = W = int(outputs.shape[1] ** 0.5)
        outputs = outputs.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return outputs


class MedSAMIBotClf(MedSAMIBot): 
    def __init__(self, *args, num_classes=2, pool_mode='cls', **kwargs): 
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(768, 2) if num_classes > 0 else nn.Identity()
        self.pool_mode = pool_mode

    def forward(self, x): 
        image_encoder_outputs = super().forward(x)

        if self.pool_mode == 'cls': 
            pre_logits = image_encoder_outputs[:, 0, :]
        elif self.pool_mode == 'patch': 
            pre_logits = image_encoder_outputs[:, self.n_cls_tokens:, :].mean(1)
        else: 
            raise ValueError()
        
        return self.head(pre_logits)


def interpolate_pos_encoding(x, pos_embed):
    npatch_in_h = x.shape[1]
    npatch_in_w = x.shape[2]

    patch_pos_embed = pos_embed

    npatch_native_h = patch_pos_embed.shape[1]
    npatch_native_w = patch_pos_embed.shape[2]

    if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
        return pos_embed

    w0 = npatch_in_w
    h0 = npatch_in_h
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.permute(0, 3, 1, 2),
        scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
        mode="bicubic",
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
    return patch_pos_embed


# monkey patching the forward pass of the image encoder to include interpolation
def image_encoder_forward_with_interpolation(self: ImageEncoderViT, x):
    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + interpolate_pos_encoding(x, self.pos_embed)

    for blk in self.blocks:
        x = blk(x)

    x = self.neck(x.permute(0, 3, 1, 2))

    return x


ImageEncoderViT.forward = image_encoder_forward_with_interpolation


class ImageEncoderViTWithClassTokenAndMasking(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        embedding_dim: int = 768,
        patch_size=16,
        img_size=1024,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self._patch_size = patch_size
        self._img_size = img_size

        self.class_token_to_image_attns = nn.ModuleList(
            [
                ClassTokenBlock(
                    emb_dim=embedding_dim, mlp_dim=embedding_dim * 4, heads=8
                )
                for _ in range(len(self.image_encoder.blocks))
            ]
        )
        self.mask_token = torch.nn.Parameter(torch.randn(embedding_dim))

    def forward(self, x, mask=None):
        x = self.image_encoder.patch_embed(x)

        if mask is not None:
            x[mask] = self.mask_token.to(x.dtype)

        x = x + self.interpolate_pos_encoding(x)

        cls_token = self.class_token.expand(x.shape[0], -1, -1)

        for blk, blk2 in zip(
            self.image_encoder.blocks, self.class_token_to_image_attns
        ):
            x = blk(x)
            cls_token = blk2(x, cls_token)

        # concatenate to typical output shape expected by vision transformers -
        # B, N, C where N is the number of patches + 1 (for the class token)
        # and class token is the first element along the N dimension
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.cat([cls_token, x], dim=1)

        return x

    def interpolate_pos_encoding(self, x):
        npatch_in_h = x.shape[1]
        npatch_in_w = x.shape[2]

        patch_pos_embed = self.image_encoder.pos_embed

        npatch_native_h = patch_pos_embed.shape[1]
        npatch_native_w = patch_pos_embed.shape[2]

        if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
            return self.image_encoder.pos_embed

        w0 = npatch_in_w
        h0 = npatch_in_h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.permute(0, 3, 1, 2),
            scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        return patch_pos_embed

    def get_image_features(self, image):
        x = self.image_encoder.patch_embed(image)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed
        for blk in self.image_encoder.blocks:
            x = blk(x)
        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2)
        return x


class ClassTokenBlock(nn.Module):
    def __init__(self, emb_dim, mlp_dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, emb_dim)
        )

    def forward(self, image, cls_token):
        shortcut = cls_token
        B, H, W, C = image.shape
        image = image.reshape(B, H * W, C)
        cls_token = self.attn(cls_token, image, image)[0]
        cls_token = cls_token + shortcut
        cls_token = cls_token + self.mlp(cls_token)
        return cls_token



class AttentionWithClsTokenWrapper(nn.Module): 
    def __init__(self, module): 
        super().__init__()
        self.module = module

    def forward(self, x, clstoken=None):
        return attn_forward_with_clstoken(self.module, x, clstoken)


class AttentionWithClsToken(Attention):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)


    def forward(self, x, clstoken=None): 
        if clstoken:
            x, clstoken = attn_forward_with_clstoken(x, clstoken)
        else: 
            x = attn_forward_with_clstoken(x)
        
    
# Another wrapper for running forward with class token on medsam model
def attn_forward_with_clstoken(self: Attention, x, clstoken=None):
    """Runs the attention forward pass with optionally including the class token.

    Args:
        self: Attention module
        x: N, H, W, C feature map to be passed through the module. Here, N could be
            either batch_size, in which case H and W are the feature map size, OR
            N could be batch_size * num_windows, in which case H and W and the window size
            (if using windowed self-attention).
        clstoken: optional cls token to use with self-attention. If it is passed,
            the class token will be added to the patch tokens in the forward pass, and the
            function will return the x and clstoken. If it is None, it will have the
            same behavior as the original function before wrapping.
    """

    B, H, W, _ = x.shape

    # qkv with shape (3, B, nHead, H * W, C)

    x = x.reshape(B, H * W, -1)  # B N_tokens D

    if clstoken is not None:
        # attach clstoken to patch tokens
        x = torch.cat([clstoken, x], dim=1)
        n_class_tokens = clstoken.shape[1]

    n_tokens = x.shape[1]
    n_patches = H * W

    qkv = (
        self.qkv(x)
        .reshape(
            B,
            n_tokens,
            3,
            self.num_heads,
            -1,
        )
        .permute(2, 0, 3, 1, 4)
    )

    # q, k, v with shape (B * nHead, H * W, C)
    q, k, v = qkv.reshape(3, B * self.num_heads, n_tokens, -1).unbind(0)

    attn = (q * self.scale) @ k.transpose(-2, -1)

    if self.use_rel_pos:
        attn[:, -n_patches:, -n_patches:] = add_decomposed_rel_pos(
            attn[:, -n_patches:, -n_patches:],
            q[:, -n_patches:, :],
            self.rel_pos_h,
            self.rel_pos_w,
            (H, W),
            (H, W),
        )

    attn = attn.softmax(dim=-1)
    x = attn @ v
    x = x.reshape(B, self.num_heads, n_tokens, -1)
    x = x.permute(0, 2, 1, 3).reshape(B, n_tokens, -1)  # B, Ntokens, head_dim * emb_dim
    x = self.proj(x)

    if clstoken is not None:
        clstoken = x[:, :n_class_tokens, :]
    x = x[:, -n_patches:, :].reshape(B, H, W, -1)

    if clstoken is None:
        return x
    else:
        return x, clstoken


class BlockWithClsTokenWrapper(nn.Module): 
    def __init__(self, module): 
        super().__init__()
        self.module = module

    def forward(self, x, clstoken=None): 
        return block_forward_with_clstoken(self.module, x, clstoken=clstoken)


class BlockWithClsToken(Block): 
    def forward(self, x, clstoken=None): 
        ...



def block_forward_with_clstoken(self: Block, x, clstoken=None):
    B, H, W, D = x.shape

    # concatenate x and clstoken for norm and shortcut
    x = einops.rearrange(x, "b h w d -> b (h w) d")
    if clstoken is not None:
        x = torch.cat([clstoken, x], dim=1)
        n_class_tokens = clstoken.shape[1]

    shortcut = x
    x = self.norm1(x)

    # Window partition
    if clstoken is not None:
        clstoken = x[:, :n_class_tokens, :]
        x = x[:, n_class_tokens:, :]

    x = einops.rearrange(x, "b (h w) d -> b h w d", h=H, w=W)
    if self.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)

        if clstoken is not None:
            n_windows = x.shape[0] // B
            clstoken = clstoken.repeat_interleave(n_windows, 0)

    if clstoken is not None:
        x, clstoken = self.attn(x, clstoken)
    else:
        x = self.attn(x, clstoken)

    # Reverse window partition
    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        # to reverse window partition on the class token, we take
        # the mean across the class token for each window.
        if clstoken is not None:
            clstoken = clstoken.reshape(B, n_windows, n_class_tokens, -1).mean(1)

    # concatenate x and clstoken
    x = einops.rearrange(x, "b h w d -> b (h w) d")
    if clstoken is not None:
        x = torch.cat([clstoken, x], dim=1)
    x = shortcut + x
    x = x + self.mlp(self.norm2(x))

    # unconcatenate them again
    if clstoken is not None:
        clstoken = x[:, :n_class_tokens, :]
        x = x[:, n_class_tokens:, :].reshape(B, H, W, D)
    else:
        x = x.reshape(B, H, W, D)

    if clstoken is None:
        return x
    else:
        return x, clstoken


class ImageEncoderViTWithClassTokenAndMaskingV2(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        embedding_dim: int = 768,
        patch_size=16,
        img_size=1024,
        use_class_token=True,
        n_class_tokens=1, 
        output_mode='all_tokens'
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.output_mode = output_mode
        self._patch_size = patch_size
        self._img_size = img_size
        self.mask_token = torch.nn.Parameter(torch.randn(embedding_dim))
        if use_class_token:
            self.class_token = nn.Parameter(
                torch.randn(1, n_class_tokens, embedding_dim)
            )
        else:
            self.class_token = None

        # new_blocks = []
        # for block in self.image_encoder.blocks: 
        #     block.attn = AttentionWithClsTokenWrapper(block.attn)
        #     block = BlockWithClsTokenWrapper(block)
        #     new_blocks.append(block)
        # self.image_encoder.blocks = nn.ModuleList(new_blocks)

        Block.forward = block_forward_with_clstoken
        Attention.forward = attn_forward_with_clstoken

    def forward(self, x, mask=None):
        x = self.image_encoder.patch_embed(x)

        if mask is not None:
            x[mask] = self.mask_token.to(x.dtype)

        x = x + self.interpolate_pos_encoding(x)
        cls_token = self.class_token.expand(x.shape[0], -1, -1)

        for blk in self.image_encoder.blocks:
            if cls_token is not None:
                x, cls_token = blk(x, cls_token)
            else:
                x = blk(x, cls_token)

        if self.output_mode == 'all_tokens':
            # concatenate to typical output shape expected by vision transformers -
            # B, N, C where N is the number of patches + 1 (for the class token)
            # and class token is the first element along the N dimension
            
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)

            if self.class_token is not None:
                x = torch.cat([cls_token, x], dim=1)
            return x

        else:
            x = x.permute(0, 3, 1, 2) # B H W C -> B C H W
            x = self.image_encoder.neck(x)
            return x

    def interpolate_pos_encoding(self, x):
        npatch_in_h = x.shape[1]
        npatch_in_w = x.shape[2]

        patch_pos_embed = self.image_encoder.pos_embed

        npatch_native_h = patch_pos_embed.shape[1]
        npatch_native_w = patch_pos_embed.shape[2]

        if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
            return self.image_encoder.pos_embed

        w0 = npatch_in_w
        h0 = npatch_in_h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.permute(0, 3, 1, 2),
            scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        return patch_pos_embed

    def get_image_features(self, image):
        x = self.image_encoder.patch_embed(image)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed
        for blk in self.image_encoder.blocks:
            x = blk(x)
        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2)
        return x


if __name__ == "__main__":
    model = build_medsam()

    model.image_encoder.neck = torch.nn.Identity()
    inp = torch.randn(1, 3, 512, 512)
    # out1 = model.image_encoder(inp).permute(0, 2, 3, 1).reshape(1, 32*32, -1)

    image_encoder = ImageEncoderViTWithClassTokenAndMaskingV2(
        model.image_encoder, use_class_token=True, n_class_tokens=2
    )
    out2 = image_encoder(inp)
    print(out2.shape)
