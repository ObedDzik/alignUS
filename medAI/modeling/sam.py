"""
Implements wrappers and registry for Segment Anything Model (SAM) models.
"""

from functools import partial
import os
from warnings import warn

import torch
from torch import nn

import sys


from external_libs.medsam.segment_anything.modeling.image_encoder import (
    Attention,
    Block,
    ImageEncoderViT,
    MLPBlock,
)

from external_libs.medsam.segment_anything.build_sam import (
    sam_model_registry as medsam_model_registry,
)

# from ._medsam.segment_anything.build_sam import sam_model_registry as medsam_model_registry
# from external_libs.sam_med2d.segment_anything import sam_model_registry as sammed_model_registry
# from huggingface_hub import PyTorchModelHubMixin

from argparse import Namespace

from medAI.modeling.segment_anything.modeling import mask_decoder
from .registry import register_model

# from timm.models.registry import register_model, generate_default_cfgs
# from timm.models._pretrained import PretrainedCfg
# from timm.models._builder import build_model_with_cfg


CHECKPOINT_DIR = os.environ.get(
    "MEDSAM_CHECKPOINT_DIR"
)  # top level checkpoint directory
if CHECKPOINT_DIR is None:
    warn(
        """Environment variable CHECKPOINT_DIR must be set. It should be a directory with sam and medsam checkpoints."""
    )


def build_sam():
    """Builds the sam-vit-b model."""
    checkpoint = os.path.join(CHECKPOINT_DIR, "sam_vit_b_01ec64.pth")
    model = medsam_model_registry["vit_b"](checkpoint=checkpoint)
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_medsam(**_):
    """
    Builds the MedSAM model by building the SAM model and loading the medsam checkpoint.
    """
    checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth")
    model = medsam_model_registry["vit_b"]()
    type(model.image_encoder).forward = forward_return_features
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_sammed_2d():
    args = Namespace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.image_size = 256
    args.encoder_adapter = True
    args.sam_checkpoint = os.path.join(CHECKPOINT_DIR, "sam-med2d_b.pth")
    model = sammed_model_registry["vit_b"](args).to(device)
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_adapter_sammed_2d():
    model = build_sammed_2d()
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


def build_adapter_sam():
    model = build_sam()
    model.image_encoder = wrap_image_encoder_with_adapter(
        model.image_encoder, adapter_dim=256
    )
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


def _load_encoder_ibot_checkpoint(encoder, checkpoint_path, config_path):
    path = checkpoint_path
    cfg_path = config_path
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)

    from medAI.factories.ibot.models import get_model, get_wrapped_models

    models = get_wrapped_models(cfg, checkpoint_path=path)
    image_encoder_pretrained = models[
        "teacher"
    ].backbone.image_encoder_wrapped.image_encoder
    msg = encoder.load_state_dict(image_encoder_pretrained.state_dict(), strict=False)

    print(f"Checkpoint path: {path}")
    print(f"Config path: {cfg_path}")
    print(f"Loading image encoder checkpoint with msg: {msg}")


_encoder_load_fns = {
    "ibot": _load_encoder_ibot_checkpoint,
}


def _build_medsam(
    sam_checkpoint="medsam",
    encoder_checkpoint=None,
    encoder_load_fn=None,
    encoder_load_kw={},
    **kwargs,
):
    """
    Builds the MedSAM model by building the SAM model and loading the medsam checkpoint.
    """
    from ._medsam.segment_anything.modeling.mask_decoder import MaskDecoder

    if sam_checkpoint == "medsam":
        checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth")
    else: 
        checkpoint = sam_checkpoint

    mask_decoder = partial(MaskDecoder, **kwargs.get("mask_decoder_kw", {}))

    model = medsam_model_registry["vit_b"](
        checkpoint=checkpoint, MaskDecoder=mask_decoder
    )

    # type(model.image_encoder).forward = forward_return_features
    if encoder_load_fn is not None:
        if isinstance(encoder_load_fn, str):
            if encoder_load_fn in _encoder_load_fns:
                encoder_load_fn = _encoder_load_fns[encoder_load_fn]
            else:
                raise ValueError(
                    f"Unknown encoder_load_fn: {encoder_load_fn}. Must be one of {list(_encoder_load_fns.keys())}"
                )

        encoder_load_fn(model.image_encoder, **encoder_load_kw)

    if encoder_checkpoint is not None:
        sd = torch.load(encoder_checkpoint, map_location="cpu")
        print(
            f"Loading image encoder checkpoint with msg: {model.image_encoder.load_state_dict(sd, strict=False)}"
        )

    return model


@register_model
def medsam(encoder_checkpoint=None, **kwargs):
    """
    Builds the MedSAM model by building the SAM model and loading the medsam checkpoint.
    """
    model = _build_medsam(encoder_checkpoint=encoder_checkpoint, **kwargs)
    wrap_with_interpolated_pos_embedding_(model)

    return model


# @register_model
# def medsam_adapter_256(encoder_checkpoint=None):
#     checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth")

#     from ._medsam.segment_anything.modeling.mask_decoder import MaskDecoder

#     # model = medsam_model_registry["vit_b"](
#     #     checkpoint=checkpoint, MaskDecoder=MaskDecoder
#     # )
#     model = medsam_model_registry["vit_b"](
#         checkpoint=checkpoint
#     )
#     if encoder_checkpoint:
#         sd = torch.load(encoder_checkpoint, map_location="cpu")
#         print(
#             f"Loading image encoder checkpoint with msg: {model.image_encoder.load_state_dict(sd, strict=False)}"
#         )

#     model.image_encoder = wrap_image_encoder_with_adapter(
#         model.image_encoder, adapter_dim=256
#     )
#     freeze_non_adapter_layers(model.image_encoder)
#     # wrap_with_interpolated_pos_embedding_(model)
#     return model



@register_model
def medsam_adapter_256(encoder_checkpoint=None):

    model = _build_medsam(encoder_checkpoint=encoder_checkpoint)
    model.image_encoder = wrap_image_encoder_with_adapter(
        model.image_encoder, adapter_dim=256
    )
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


@register_model
def medsam_adapter(
    encoder_checkpoint=None,
    adapter_dim=256,
    **kwargs,
):

    model = _build_medsam(encoder_checkpoint=encoder_checkpoint, **kwargs)
    model.image_encoder = wrap_image_encoder_with_adapter(
        model.image_encoder, adapter_dim=adapter_dim
    )
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


@register_model
def medsam_adapter_upsample(encoder_checkpoint=None, adapter_dim=256):

    from ._medsam.segment_anything.modeling.mask_decoder import MaskDecoder

    mask_decoder = partial(MaskDecoder, num_extra_upscale_layers=2)

    model = medsam_model_registry["vit_b"](
        checkpoint=checkpoint, MaskDecoder=mask_decoder
    )
    if encoder_checkpoint:
        sd = torch.load(encoder_checkpoint, map_location="cpu")
        print(
            f"Loading image encoder checkpoint with msg: {model.image_encoder.load_state_dict(sd, strict=False)}"
        )

    model.image_encoder = wrap_image_encoder_with_adapter(
        model.image_encoder, adapter_dim=adapter_dim
    )
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


@register_model
def medsam_image_encoder(**kwargs):
    medsam = build_medsam(**kwargs)
    return medsam.image_encoder


@register_model
def sam(**kwargs):
    return build_sam()


@register_model
def sammed_2d(**kwargs):
    return build_sammed_2d()


@register_model
def sam_adapter_256(**kwargs):
    return build_adapter_sam()


@register_model
def sammed_2d_adapter_256(**kwargs):
    return build_adapter_sammed_2d()


@register_model
def sam_unprompted_mask_prediction(**kwargs):
    return SAMForUnpromptedMaskPrediction()


class Adapter(nn.Module):
    def __init__(self, feature_dim, adapter_dim, init_scale=1e-3):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(feature_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, feature_dim)
        self.act = nn.GELU()

        # initializations to make it close to identity function
        nn.init.uniform_(self.down_project.weight, -init_scale, init_scale)
        nn.init.uniform_(self.up_project.weight, -init_scale, init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return self.up_project(self.act(self.down_project(x))) + x


class AdapterAttn(nn.Module):
    def __init__(self, attn: Attention, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()
        self.attn = attn
        embedding_dim = attn.proj.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.adapter(x)
        return x


class AdapterMLPBlock(nn.Module):
    def __init__(self, mlp: MLPBlock, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()

        self.mlp = mlp
        embedding_dim = mlp.lin1.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.adapter(x)
        return x


#These three are the medsam_adapter ssl training.
# class Adapter(nn.Module):
#     def __init__(self, feature_dim, adapter_dim, init_scale=1e-3):
#         super(Adapter, self).__init__()
#         self.down_project = nn.Linear(feature_dim, adapter_dim)
#         self.up_project = nn.Linear(adapter_dim, feature_dim)
#         self.act = nn.GELU()

#         # Initialize to identity-like
#         nn.init.uniform_(self.down_project.weight, -init_scale, init_scale)
#         nn.init.uniform_(self.up_project.weight, -init_scale, init_scale)
#         nn.init.zeros_(self.down_project.bias)
#         nn.init.zeros_(self.up_project.bias)

#     def forward(self, x):
#         # Ensure x is (B, N, C)
#         if x.ndim == 4:  # (B, C, H, W)
#             x = x.permute(0, 2, 3, 1).flatten(1, 2)

#         # Adapter residual
#         return x + self.up_project(self.act(self.down_project(x)))  # output dim = feature_dim


# class AdapterAttn(nn.Module):
#     def __init__(self, attn: nn.Module, adapter_dim: int, init_scale: float = 1e-3):
#         super().__init__()
#         self.attn = attn
#         embedding_dim = attn.proj.out_features
#         self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

#     def forward(self, x, clstoken=None):
#         # Call the original (now monkey-patched) attention
#         result = self.attn(x, clstoken)
        
#         # Handle different return types
#         if clstoken is not None:
#             x, clstoken = result
#             # Apply adapter to patches
#             if x.ndim == 4:  # (B, H, W, C)
#                 B, H, W, C = x.shape
#                 x = x.reshape(B, H * W, C)
#                 x = self.adapter(x)
#                 x = x.reshape(B, H, W, C)
#             else:  # (B, N, C)
#                 x = self.adapter(x)
            
#             # Apply adapter to class tokens
#             clstoken = self.adapter(clstoken)
#             return x, clstoken
#         else:
#             x = result
#             # Apply adapter
#             if x.ndim == 4:  # (B, H, W, C)
#                 B, H, W, C = x.shape
#                 x = x.reshape(B, H * W, C)
#                 x = self.adapter(x)
#                 x = x.reshape(B, H, W, C)
#             else:  # (B, N, C)
#                 x = self.adapter(x)
#             return x


# class AdapterMLPBlock(nn.Module):
#     def __init__(self, mlp: nn.Module, adapter_dim: int, init_scale: float = 1e-3):
#         super().__init__()
#         self.mlp = mlp
#         if hasattr(mlp, 'lin2'):
#             embedding_dim = mlp.lin2.out_features
#         else:
#             embedding_dim = mlp.lin1.in_features
#         self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

#     def forward(self, x):
#         x = self.mlp(x)
#         x = self.adapter(x)
#         return x


def wrap_block_with_adapter(block: Block, adapter_dim: int, init_scale: float = 1e-3):
    block.attn = AdapterAttn(block.attn, adapter_dim, init_scale=init_scale)
    block.mlp = AdapterMLPBlock(block.mlp, adapter_dim, init_scale=init_scale)
    return block

def wrap_image_encoder_with_adapter(
    image_encoder: ImageEncoderViT, adapter_dim: int, init_scale: float = 1e-3
):
    new_blocks = torch.nn.ModuleList()
    for block in image_encoder.blocks:
        new_block = wrap_block_with_adapter(block, adapter_dim, init_scale=init_scale)
        new_blocks.append(new_block)

    image_encoder.blocks = new_blocks

    return image_encoder


def freeze_non_adapter_layers(model: nn.Module):
    for name, param in model.named_parameters():
        if "adapter" not in name.lower():
            param.requires_grad = False

    return model


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

#TODO: THIS CODE IS THE ORIGINAL
def forward_return_features(image_encoder: ImageEncoderViT, x, return_hiddens=False):
    # "Return hiddens" feature added

    x = image_encoder.patch_embed(x)
    if image_encoder.pos_embed is not None:
        x = x + interpolate_pos_encoding(x, image_encoder.pos_embed)

    hiddens = []
    for blk in image_encoder.blocks:
        x = blk(x)
        if return_hiddens:
            hiddens.append(x)

    x = image_encoder.neck(x.permute(0, 3, 1, 2))

    return (x, hiddens) if return_hiddens else x

# #TODO: THIS IS FROM TAREK
# def forward_return_features(image_encoder: ImageEncoderViT, x, return_hiddens=False):
#     # "Return hiddens" feature added
#     x = image_encoder.patch_embed(x)
#     if image_encoder.pos_embed is not None:
#         x = x + interpolate_pos_encoding(x, image_encoder.pos_embed)
#     hiddens = []
#     for blk in image_encoder.blocks:
#         x = blk(x)
#         if return_hiddens:
#             hiddens.append(x)
#     x=x.permute(0, 3, 1, 2)
#     x_neck = image_encoder.neck(x)

#     return (x, hiddens) if return_hiddens else x_neck,x


def wrap_with_interpolated_pos_embedding_(sam_model):
    type(sam_model.image_encoder).forward = forward_return_features


class SAMForUnpromptedMaskPrediction(nn.Module):
    def __init__(self, sam_variant: str = "sam", dropout=0.1):
        super().__init__()
        self.sam = get_sam_variant(sam_variant)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_image = x
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.sam.image_encoder(x)
        features = self.dropout(features)
        sparse_embedding, dense_embedding = self.sam.prompt_encoder(None, None, None)

        B, C, H, W = features.shape

        pos_embedding = self.sam.prompt_encoder.get_dense_pe()
        if pos_embedding.shape[2:] != features.shape[2:]:
            pos_embedding = torch.nn.functional.interpolate(
                pos_embedding,
                size=features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        if dense_embedding.shape[2:] != features.shape[2:]:
            dense_embedding = torch.nn.functional.interpolate(
                dense_embedding,
                size=features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        low_res_mask_logits = self.sam.mask_decoder.forward(
            features,
            pos_embedding,
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
        )[0]

        mask_logits = torch.nn.functional.interpolate(
            low_res_mask_logits,
            size=input_image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return mask_logits
