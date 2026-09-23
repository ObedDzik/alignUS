from functools import partial
import os
import sys
from warnings import warn
import loralib as lora


DINOV3_LIBRARY_PATH = os.getenv(
    "DINOV3_LIBRARY_PATH",
)
if DINOV3_LIBRARY_PATH is not None:
    sys.path.append(DINOV3_LIBRARY_PATH)

DINOV3_CHECKPOINTS_PATH = os.getenv(
    "DINOV3_CHECKPOINTS_PATH",
)
if DINOV3_CHECKPOINTS_PATH is None:
    warn(
        "DINOV3_CHECKPOINTS_PATH environment variable not set. Set this environment variable to the path where DINOv3 checkpoints are stored.",
    )
msda_ops_dir = os.getenv("DINOV3_MSDA_OPS_DIR")
if msda_ops_dir is not None:
    sys.path.append(msda_ops_dir)

try:
    import dinov3
    import dinov3.hub
    import dinov3.hub.backbones
    from dinov3.eval.segmentation.models import FeatureDecoder
    from dinov3.eval.segmentation.models.heads.linear_head import LinearHead
    from dinov3.layers import (
        Mlp,
        SwiGLUFFN,
    )  # SelfAttentionBlock is used via model.blocks
except ImportError:
    warn(
        "DINOv3 library not found. Please specify the DINOV3_LIBRARY_PATH environment variable pointing to the DINOv3 library location.",
        ImportWarning,
    )
    dinov3 = None

from omegaconf import OmegaConf
import torch


from medAI.modeling.registry import create_model, register_model
import os
import tempfile
from torch import nn


@register_model
def dinov3_segmentator(dinov3_name="dinov3_vitl16"):

    # model = torch.hub.load('facebookresearch/dinov3', dinov3_name, pretrained=True)

    model = dinov3.hub.backbones.__dict__[dinov3_name](pretrained=True)

    from dinov3.eval.segmentation.models import build_segmentation_decoder

    model = build_segmentation_decoder(
        model, "dinov3_vitl16", "m2f", autocast_dtype=torch.float16
    ).cuda()

    return model


@register_model
def dinov3_ssl_finetune(*, config_path, checkpoint_path):
    from dinov3.models import build_model_from_cfg
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        from torch.distributed.checkpoint import format_utils as fmt

        with tempfile.NamedTemporaryFile() as tmpfile:
            print(
                "Converting distributed checkpoint to",
                tmpfile.name,
                ". This could take a while...",
            )
            fmt.dcp_to_torch_save(checkpoint_path, tmpfile.name)
            state_dict = torch.load(tmpfile.name, weights_only=False)
            # model = build_model_for_eval(cfg, tmpfile.name)

    model = build_model_from_cfg(cfg)[0]
    model_state_dict = state_dict["model"]
    model_state_dict = {
        k.replace("student.", ""): v
        for k, v in model_state_dict.items()
        if k.startswith("student.")
    }
    model_state_dict = {
        k.replace("backbone.", ""): v
        for k, v in model_state_dict.items()
        if k.startswith("backbone.")
    }
    model.load_state_dict(model_state_dict, strict=False, assign=True)

    return model


class DinoV3DenseFeaturesWithLinearHead(nn.Module):
    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.conv = nn.Conv2d(backbone.embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x, reshape=True)[-1]
        return self.conv(features)


@register_model
def dinov3_dense_features_with_linear_head(
    *, backbone_cfg=dict(name="dinov3_backbone"), freeze_backbone=False, **kwargs
):
    backbone = create_model(**backbone_cfg)
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    return DinoV3DenseFeaturesWithLinearHead(backbone, **kwargs)


@register_model
def dinov3_backbone(name="dinov3_vitl16", wrapper=None, lora=False):
    import dinov3.hub.backbones

    backbone = dinov3.hub.backbones.__dict__[name](pretrained=True)

    if lora:
        import loralib as lora

    if wrapper:
        if isinstance(wrapper, str):
            wrapper = wrappers_dict[wrapper]
        backbone = wrapper(backbone)

    return backbone


@register_model
def dinov3_backbone_from_checkpoint(config_path, checkpoint_path, wrapper=None):
    from dinov3.models import build_model
    from dinov3.configs import get_default_config

    default_cfg = get_default_config()
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(config_path))

    student, teacher, embed_dim = build_model(cfg.student)

    import os
    print("Loading:", checkpoint_path, "Exists:", os.path.exists(checkpoint_path))


    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt = ckpt["teacher"]
    ckpt = {
        k.replace("backbone.", ""): v
        for k, v in ckpt.items()
        if k.startswith("backbone.")
    }
    print(teacher.load_state_dict(ckpt, strict=True))

    if wrapper:
        if isinstance(wrapper, str):
            wrapper = wrappers_dict[wrapper]
        teacher = wrapper(teacher)

    return teacher


@register_model
def dinov3_m2f_segmentor_for_binary_mask_prediction():
    from dinov3.hub.segmentors import _make_dinov3_m2f_segmentor
    from dinov3.eval.segmentation.models import (
        build_segmentation_decoder,
        FeatureDecoder,
    )
    from dinov3.hub.backbones import dinov3_vitl16

    backbone = dinov3_vitl16(pretrained=True)

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            model_output = self.model(x)
            return model_output["pred_masks"][:, 0:1, :, :]  # return only first channel

    decoder = build_segmentation_decoder(
        backbone_model=backbone,
        backbone_name="dinov3_vitl16",
        decoder_type="m2f",
        num_classes=10,
    )

    return Wrapper(decoder)


@register_model
def dinov3_linear_segmentor_for_binary_mask_prediction(
    *, backbone_name="dinov3_vitl16"
):
    from dinov3.hub import backbones
    from dinov3.eval.segmentation.models import build_segmentation_decoder

    backbone = backbones.__dict__[backbone_name](pretrained=True)
    decoder = build_segmentation_decoder(
        backbone_model=backbone,
        decoder_type="linear",
        num_classes=1,
    )

    return decoder


@register_model
def dinov3_simple_segmentor_for_binary_mask_prediction(
    *, backbone_cfg: dict = {}, backbone_name=None, dropout=0.1
):
    if backbone_name is not None:
        from dinov3.hub import backbones

        backbone = backbones.__dict__[backbone_name](pretrained=True)
    else:
        backbone = create_model(**backbone_cfg)
    from dinov3.eval.utils import ModelWithIntermediateLayers
    from dinov3.eval.segmentation.models import _get_backbone_out_indices

    out_indices = _get_backbone_out_indices(backbone)
    autocast_ctx = partial(
        torch.autocast, device_type="cuda", enabled=True, dtype=torch.float16
    )

    wrapped_backbone = DINOV3BackboneWrapperForFeatureMapsList(backbone, n=out_indices)

    embed_dim = wrapped_backbone.backbone.embed_dim
    embed_dim = [embed_dim] * len(out_indices)
    decoder = LinearHead(
        in_channels=embed_dim,
        n_output_channels=1,
        dropout=dropout,
    )

    segmentation_model = FeatureDecoder(
        torch.nn.ModuleList(
            [
                wrapped_backbone,
                decoder,
            ]
        ),
        autocast_ctx=autocast_ctx,
    )
    return segmentation_model


"""
import torch

REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>

# DINOv3 ViT models pretrained on web images
dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

# DINOv3 ConvNeXt models pretrained on web images
dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_small = torch.hub.load(REPO_DIR, 'dinov3_convnext_small', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_base = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_large = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

# DINOv3 ViT models pretrained on satellite imagery
dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

"""


@register_model
def dinov3_vitl16(**kwargs):
    import torch

    REPO_DIR = DINOV3_LIBRARY_PATH
    weight_basename = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    # DINOV3_CHECKPOINTS_PATH='/datasets/exactvu_pca/checkpoint_store'
    # REPO_DIR='/home/obed/projects/aip-medilab/obed/medproj/dinov3'

    weights = os.path.join(DINOV3_CHECKPOINTS_PATH, weight_basename)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {weights}")
    kwargs["weights"] = weights
    model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", **kwargs)
    return model


@register_model
def dinov3_vith16plus(**kwargs):
    import torch

    REPO_DIR = DINOV3_LIBRARY_PATH
    weight_basename = "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    weights = os.path.join(DINOV3_CHECKPOINTS_PATH, weight_basename)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {weights}")
    kwargs["weights"] = weights

    model = torch.hub.load(REPO_DIR, "dinov3_vith16plus", source="local", **kwargs)
    return model


class DINOV3BackboneWrapperForFeatureMaps(nn.Module):
    """Wrapper around DINOv3 backbone to extract feature maps of shape (B, C, H, W)"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        features = self.backbone.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]
        B, N, C = patch_tokens.shape
        H = W = int(N**0.5)
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return patch_tokens


class DINOV3BackboneWrapperForFeatureMapsList(nn.Module):
    def __init__(self, backbone, n=None):
        super().__init__()
        if n is None:
            n = len(backbone.blocks)
        self.n = n
        self.backbone = backbone

    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x, n=self.n, reshape=True)
        return features
    
    def forward_with_final(self, x):
        final = self.backbone.forward_features(x)  # gets the full forward pass with norm
        return final


@register_model
def dinov3_backbone_wrapper_for_feature_maps(*, backbone_cfg: dict):
    backbone = create_model(**backbone_cfg)
    return DINOV3BackboneWrapperForFeatureMaps(backbone)


@register_model
def dinov3_backbone_wrapper_for_feature_maps_list(*, backbone_cfg: dict):
    backbone = create_model(**backbone_cfg)
    return DINOV3BackboneWrapperForFeatureMapsList(backbone)


wrappers_dict = {
    "feature_maps": DINOV3BackboneWrapperForFeatureMaps,
    "feature_maps_list": DINOV3BackboneWrapperForFeatureMapsList,
}


def replace_linear_with_lora(old_linear: torch.nn.Linear, **lora_kw) -> lora.Linear:
    """Create a LoRA-wrapped Linear with the same weights/bias as old_linear."""
    new_linear = lora.Linear(
        old_linear.in_features,
        old_linear.out_features,
        **lora_kw,
        bias=old_linear.bias is not None,
    )
    # Copy base weights so the behavior is initially identical
    new_linear.weight.data.copy_(old_linear.weight.data)
    if old_linear.bias is not None:
        new_linear.bias.data.copy_(old_linear.bias.data)
    return new_linear


def add_lora_to_dinov3_vit(
    model,
    r=8,
    lora_alpha=1.0,
    lora_on_mlp=False,
    bias="none",
    apply_lora_to_patch_embed=False,
    make_norm_trainable=False,
):
    """
    model: DinoVisionTransformer
    r: LoRA rank
    lora_on_mlp: whether to also LoRA-ize the FFN linears
    """

    if apply_lora_to_patch_embed:
        old_patch_embed = model.patch_embed.proj
        new_patch_embed = lora.Conv2d(

            old_patch_embed.in_channels,
            old_patch_embed.out_channels,
            kernel_size=old_patch_embed.kernel_size[0],
            stride=old_patch_embed.stride,
            padding=old_patch_embed.padding,
            bias=old_patch_embed.bias is not None,
            r=r,
            lora_alpha=lora_alpha,
        )
        # Copy base weights so the behavior is initially identical
        new_patch_embed.conv.weight.data.copy_(old_patch_embed.weight.data)
        if old_patch_embed.bias is not None:
            new_patch_embed.conv.bias.data.copy_(old_patch_embed.bias.data)
        model.patch_embed.proj = new_patch_embed

    for blk in model.blocks:  # each blk is a SelfAttentionBlock
        # --- Attention qkv ---
        old_qkv = blk.attn.qkv
        blk.attn.qkv = replace_linear_with_lora(old_qkv, r=r, lora_alpha=lora_alpha)

        # --- Attention proj ---
        old_proj = blk.attn.proj
        blk.attn.proj = replace_linear_with_lora(old_proj, r=r, lora_alpha=lora_alpha)

        if lora_on_mlp:
            mlp = blk.mlp
            if isinstance(mlp, Mlp):
                # Two-linears FFN: fc1, fc2
                mlp.fc1 = replace_linear_with_lora(mlp.fc1, r=r, lora_alpha=lora_alpha)
                mlp.fc2 = replace_linear_with_lora(mlp.fc2, r=r, lora_alpha=lora_alpha)
            elif isinstance(mlp, SwiGLUFFN):
                # SwiGLUFFN uses w1, w2, w3
                mlp.w1 = replace_linear_with_lora(mlp.w1, r=r, lora_alpha=lora_alpha)
                mlp.w2 = replace_linear_with_lora(mlp.w2, r=r, lora_alpha=lora_alpha)
                mlp.w3 = replace_linear_with_lora(mlp.w3, r=r, lora_alpha=lora_alpha)

    lora.mark_only_lora_as_trainable(model, bias=bias)

    if make_norm_trainable:
        for n, p in model.named_parameters():
            if "norm" in n:
                p.requires_grad = True

    for n, p in model.named_parameters():
        print(f"{n}: requires_grad={p.requires_grad}")


@register_model
def dinov3_lora(*, backbone_cfg: dict, **lora_kwargs):
    backbone = create_model(**backbone_cfg)
    add_lora_to_dinov3_vit(backbone, **lora_kwargs)
    return backbone
