"""
Implements wrappers and registry for Segment Anything Model (SAM) models.
"""

import logging
import os
import urllib.request
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Literal
from warnings import warn

import einops
import torch
from torch import nn

from medAI.layers.adapter import Adapter, freeze_non_adapter_layers

from ._medsam.segment_anything.build_sam import sam_model_registry
from ._medsam.segment_anything.modeling.image_encoder import Attention, Block
from ._medsam.segment_anything.modeling.image_encoder import (
    ImageEncoderViT as _ImageEncoderViT,
)
from ._medsam.segment_anything.modeling.image_encoder import (
    MLPBlock,
    add_decomposed_rel_pos,
    window_partition,
    window_unpartition,
)
from ._medsam.segment_anything.modeling.mask_decoder import MaskDecoder
from ._medsam.segment_anything.modeling.prompt_encoder import PromptEncoder
from ._medsam.segment_anything.modeling.sam import Sam
from ._medsam.segment_anything.modeling.transformer import TwoWayTransformer
from .registry import create_model, register_model


CHECKPOINT_DIR = os.environ.get(
    "MEDSAM_CHECKPOINT_DIR"
)  # top level checkpoint directory
if CHECKPOINT_DIR is None:
    warn(
        """Environment variable CHECKPOINT_DIR must be set. It should be a directory with sam and medsam checkpoints."""
    )


class AdapterAttn(Attention):
    def __init__(self, *args, adapter_dim=256, init_scale=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter = Adapter(self.dim, adapter_dim=adapter_dim, init_scale=init_scale)

    def forward(self, x):
        x = super().forward(x)
        x = self.adapter(x)
        return x


class AdapterMLPBlock(MLPBlock):
    def __init__(self, *args, adapter_dim=None, init_scale=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        if adapter_dim is not None:
            self.adapter = Adapter(
                self.dim, adapter_dim=adapter_dim, init_scale=init_scale
            )
        else:
            self.adapter = nn.Identity()

    def forward(self, x):
        x = super().forward(x)
        x = self.adapter(x)
        return x


class AdapterBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mlp_block_cls=AdapterMLPBlock, attn_cls=AdapterAttn, **kwargs)


class AdapterAttnWithClassToken(Attention):
    """Makes attention support the use of an additional class token (normally sam attention is patch tokens only)"""

    def __init__(self, *args, adapter_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_dim = adapter_dim
        if adapter_dim is not None:
            self.adapter = Adapter(self.dim, self.adapter_dim)
        else:
            self.adapter = nn.Identity()

    def forward(self, x, clstoken=None):
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
        x = x.permute(0, 2, 1, 3).reshape(
            B, n_tokens, -1
        )  # B, Ntokens, head_dim * emb_dim
        x = self.proj(x)

        x = self.adapter(x)

        if clstoken is not None:
            clstoken = x[:, :n_class_tokens, :]
        x = x[:, -n_patches:, :].reshape(B, H, W, -1)

        if clstoken is None:
            return x
        else:
            return x, clstoken


class BlockWithClassToken(Block):
    def __init__(
        self,
        *args,
        attn_cls=AdapterAttnWithClassToken,
        mlp_cls=AdapterMLPBlock,
        adapter_dim=None,
        **kwargs,
    ):
        attn_cls = partial(attn_cls, adapter_dim=adapter_dim)
        mlp_cls = partial(mlp_cls, adapter_dim=adapter_dim)
        super().__init__(*args, attn_cls=attn_cls, mlp_block_cls=mlp_cls, **kwargs)

    def forward(self, x, clstoken=None):
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


class ImageEncoderViT(_ImageEncoderViT):
    def __init__(
        self,
        *args,
        n_cls_tokens=0,
        output_mode: Literal["all_tokens"] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.output_mode = output_mode
        if self.output_mode == "all_tokens":
            self.neck = None

        self.n_cls_tokens = n_cls_tokens

        if n_cls_tokens > 0:
            self.class_token = nn.Parameter(
                torch.randn(1, n_cls_tokens, self.embed_dim)
            )
        else:
            self.class_token = None

        self.mask_token = torch.nn.Parameter(torch.randn(self.embed_dim))

    def forward(self, x, mask=None):
        x = self.patch_embed(x)

        if mask is not None:
            x[mask] = self.mask_token.to(x.dtype)

        x = x + self.interpolate_pos_encoding(x)

        if self.class_token is not None:
            cls_token = self.class_token.expand(x.shape[0], -1, -1)
        else:
            cls_token = None

        for blk in self.blocks:
            if cls_token is not None:
                x, cls_token = blk(x, cls_token)
            else:
                x = blk(x, cls_token)

        if self.output_mode == "all_tokens":
            # concatenate to typical output shape expected by vision transformers -
            # B, N, C where N is the number of patches + 1 (for the class token)
            # and class token is the first element along the N dimension

            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)

            if self.class_token is not None:
                x = torch.cat([cls_token, x], dim=1)
            return x

        else:
            x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
            x = self.neck(x)
            return x

    def interpolate_pos_encoding(self, x):
        npatch_in_h = x.shape[1]
        npatch_in_w = x.shape[2]

        patch_pos_embed = self.pos_embed

        npatch_native_h = patch_pos_embed.shape[1]
        npatch_native_w = patch_pos_embed.shape[2]

        if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
            return self.pos_embed

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


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    load_strict=True,
    adapter_dim=None,
    image_encoder_kw={},
    mask_decoder_kw={},
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            block_cls=partial(BlockWithClassToken, adapter_dim=adapter_dim),
            **image_encoder_kw,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            **mask_decoder_kw,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if adapter_dim is not None: 
        for name, param in sam.image_encoder.named_parameters():
            if "adapter" not in name.lower():
                param.requires_grad = False

    sam.eval()
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == "y":
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-B checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
        elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == "y":
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-H checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
        elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == "y":
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-L checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print(
            f"Loaded sam state with message {sam.load_state_dict(state_dict, strict=load_strict)}"
        )

    return sam


@register_model
def medsam_vit_b(
    medsam_checkpoint=None,
    ibot_encoder_checkpoint=None,
    medibot_encoder_checkpoint=None,
    lora=None,
    **kwargs,
):
    if medsam_checkpoint is None and CHECKPOINT_DIR is not None:
        medsam_checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth")
    
    sam = _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=medsam_checkpoint,
        load_strict=False,
        **kwargs,
    )

    if medibot_encoder_checkpoint:
        # Assumes that this was pretrained by the 'medibot' scripts
        sd = torch.load(medibot_encoder_checkpoint, map_location="cpu")
        sd = sd["teacher"]
        sd = {k.replace("vit.", ""): v for k, v in sd.items() if k.startswith("vit.")}
        sd = {
            k.replace("backbone.", ""): v
            for k, v in sd.items()
            if k.startswith("backbone.")
        }
        sd = {k.replace("image_encoder_wrapped.", ""): v for k, v in sd.items()}
        sd = {k.replace("image_encoder.", ""): v for k, v in sd.items()}

        print(
            f"Loaded image encoder from {medibot_encoder_checkpoint} with message {sam.image_encoder.load_state_dict(sd, strict=False)}"
        )
    
    if ibot_encoder_checkpoint: 
        # Assumes that this was pretrained by the 'ibot' scripts
        sd = torch.load(ibot_encoder_checkpoint, map_location="cpu")
        sd = sd["teacher"]
        sd = {
            k.replace("backbone.", ""): v
            for k, v in sd.items()
            if k.startswith("backbone.")
        }
        sd = {k.replace("image_encoder_wrapped.", ""): v for k, v in sd.items()}
        sd = {k.replace("image_encoder.", ""): v for k, v in sd.items()}

        print(
            f"Loaded image encoder from {ibot_encoder_checkpoint} with message {sam.image_encoder.load_state_dict(sd, strict=False)}"
        )

    if lora is not None: 
        logging.info(f"Apply lora...")
        import loralib
        
        mode = lora.split(":")[0]
        dim = int(lora.split(":")[1])
        
        if mode == 'qv':
            def _apply_lora(module, **kwargs):
                for name, submodule in module.named_children(): 
                    if name.endswith('qkv'):
                        new_submodule = loralib.MergedLinear(
                            submodule.in_features,
                            submodule.out_features,
                            **kwargs,
                            enable_lora=[True, False, False]
                        )
                        new_submodule.load_state_dict(submodule.state_dict(), strict=False)
                    else: 
                        new_submodule = _apply_lora(submodule, **kwargs)
                    setattr(module, name, new_submodule)
                return module
        
        elif mode == 'all': 
            def _apply_lora(module, **kwargs):
                for name, submodule in module.named_children(): 
                    if isinstance(submodule, nn.Linear):
                        new_submodule = loralib.Linear(
                            submodule.in_features,
                            submodule.out_features,
                            **kwargs
                        )
                        new_submodule.load_state_dict(submodule.state_dict(), strict=False)
                    else: 
                        new_submodule = _apply_lora(submodule, **kwargs)  
                    setattr(module, name, new_submodule)
                return module
        else: 
            raise ValueError(f"Unknown mode {mode}")    
        
        _apply_lora(sam.image_encoder, r=dim, lora_alpha=dim/4)

        loralib.mark_only_lora_as_trainable(sam.image_encoder)

    return sam


@register_model
def medsam_vit_b_image_encoder_clstoken(**kwargs): 
    print(f'`medsam_vit_b`: ignoring unused bindings {kwargs}')

    model = medsam_vit_b(
        image_encoder_kw=dict(n_cls_tokens=1, output_mode='all_tokens', **kwargs)
    )    
    return model.image_encoder


@register_model
def medsam_vit_b_image_encoder_classifier(encoder_checkpoint=None, **kwargs): 
    image_encoder = medsam_vit_b_image_encoder_clstoken(**kwargs)
    if encoder_checkpoint: 
        print(image_encoder.load_state_dict(
            torch.load(encoder_checkpoint, map_location='cpu'), 
            strict=False
        ))

    for param in image_encoder.parameters(): 
        param._is_backbone = True

    class ClassifierWrapper(nn.Module):
        def __init__(self, backbone): 
            super().__init__()
            self.backbone = backbone 
            self.fc = nn.Linear(backbone.embed_dim, 2)
        
        def forward(self, x): 
            return self.fc(self.backbone(x)[:, 0, :])

    return ClassifierWrapper(image_encoder)

