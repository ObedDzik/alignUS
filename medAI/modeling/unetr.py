from typing import Literal
from warnings import warn
import einops
import torch
from torch import nn
from monai.networks.nets.unetr import (
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from huggingface_hub import PyTorchModelHubMixin

from medAI.modeling.vision_transformer import VisionTransformer
from medAI.modeling.registry import create_model, register_model
# from projects.dinov3.dinov3.eval.detection.models import backbone

# from medAI.modeling.asymmetric_convolution_models import create_model
# from medAI.modeling.registry import register_model, register_pretrained_cfgs
# from medAI.modeling.vision_transformer import VisionTransformer


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


def _get_image_encoder(image_encoder_name):
    if image_encoder_name == "vit_small":
        from .vision_transformer import vit_small

        vit = vit_small(patch_size=16, img_size=224)
        image_encoder = VITImageEncoderWrapperForUNETR(vit)
        embedding_size = 384
    elif image_encoder_name == "vit_base":
        from .vision_transformer import vit_base

        vit = vit_base(patch_size=16, img_size=224)
        image_encoder = VITImageEncoderWrapperForUNETR(vit)
        embedding_size = 768
    elif image_encoder_name == "medsam":
        from .sam import build_medsam

        sam = build_medsam()
        image_encoder = SAMWrapperForUNETR(sam.image_encoder)
        embedding_size = 768
    elif image_encoder_name == "dino" or image_encoder_name == "dino_v2_vitb14":
        from .registry import dinov2_vitb14

        dino = dinov2_vitb14()
        image_encoder = DinoV2WrapperForUNETR(dino)
        embedding_size = 768
    else:
        raise ValueError(f"Unknown image encoder {image_encoder_name}")
    return image_encoder, embedding_size


class UNETR(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        image_encoder: str | nn.Module = "vit_small",
        backbone_out_format='bhwc',
        embedding_size=None,
        feature_size=64,
        out_channels=1,
        norm_name="instance",
        input_size=1024,
        output_size=256,
        backbone_weights_path: str | None = None,
        interaction_indices: list[int] = [3, 6, 9, 11],
    ):
        """
        Args:
            image_encoder: a vision transformer backbone model. Its call pattern should be `image_encoder(x)`.
            The return value should be a list of tensors, shaped to B, H, W, C. Only hidden states at certain
            indices will be used in the UNETR model. 12 hidden states are expected.
            embedding_size: the size of the output of the image_encoder model.
            feature_size: the number of features coming out of the fist encoder block - dimensionality of intermediate layers is
                a multiple of this.
            out_channels: the number of output channels.
            norm_name: the name of the normalization layer to use in the UNETR model.
            input_size: the size of the input image.
            output_size: the size of the output heatmap.
        """

        super().__init__()

        self.backbone_out_format = backbone_out_format
        self.interaction_indices = interaction_indices
        self.return_dict = False

        if isinstance(image_encoder, str):
            self.image_encoder, embedding_size = _get_image_encoder(image_encoder)
            self.backbone = self.image_encoder
        else:
            if embedding_size is None:
                raise ValueError(
                    "embedding_size must be provided if image_encoder is a nn.Module"
                )
            self.image_encoder = image_encoder
            self.backbone = self.image_encoder

        if backbone_weights_path is not None:
            try:
                backbone_weights = torch.load(backbone_weights_path, map_location="cpu")
                self.image_encoder.load_state_dict(backbone_weights)
            except:
                warn(
                    "Could not load backbone weights. This is expected if you are loading a model from pre-trained weights."
                )

        embedding_size = embedding_size
        feature_size = feature_size  # divides embedding size

        # if the input size is greater than the output size, we need to downsample.
        # however, we don't sample the input but rather the transformer intermediate outputs
        if input_size > output_size:
            self.downsample = torch.nn.MaxPool2d(input_size // output_size)
        else:
            self.downsample = torch.nn.Identity()

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=3,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            upsample_kernel_size=2,
            stride=1,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            num_layer=0,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out_block = UnetOutBlock(
            spatial_dims=2,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def backbone_out_to_conv_in(self, x):
        if self.backbone_out_format == 'bhwc':
            return x.permute(0, 3, 1, 2)
        elif self.backbone_out_format == 'bchw':
            return x
        else: 
            raise ValueError(f"Unknown backbone_out_format {self.backbone_out_format}")

    def forward(self, x):
        hiddens = self.image_encoder(x)
        last_encoder_features = hiddens[-1]
        final = self.image_encoder.forward_with_final(x)

        hiddens = [self.backbone_out_to_conv_in(h) for h in hiddens]

        H_fmap_base, W_fmap_base = hiddens[0].shape[-2], hiddens[0].shape[-1]
        H_im, W_im = x.shape[2], x.shape[3]

        if H_fmap_base * 2**4 != H_im or W_fmap_base * 2**4 != W_im:
            warn(
                f"""Detected input size {H_im}x{W_im} and feature map size {H_fmap_base}x{W_fmap_base}. Feature maps will be resampled to 
compatible size of {H_im // 16}x{W_im // 16}"""
            )
            hiddens_new = []
            for hidden in hiddens:
                hidden = torch.nn.functional.interpolate(
                    hidden, size=(H_im // 16, W_im // 16), mode="bilinear"
                )
                hiddens_new.append(hidden)
            hiddens = hiddens_new

        x1 = self.downsample(x)
        x2 = self.downsample(hiddens[self.interaction_indices[0]])
        x3 = self.downsample(hiddens[self.interaction_indices[1]])
        x4 = self.downsample(hiddens[self.interaction_indices[2]])
        x5 = self.downsample(hiddens[self.interaction_indices[3]])

        enc1 = self.encoder1(x1)
        enc2 = self.encoder2(x2)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)
        dec4 = x5

        dec3 = self.decoder1(dec4, enc4)
        dec2 = self.decoder2(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder4(dec1, enc1)

        if self.return_dict:
            return dict(cancer_logits=self.out_block(dec0), image_feats=last_encoder_features, final_feats=final)
        else:
            return self.out_block(dec0)


class VITImageEncoderWrapperForUNETR(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        hiddens = self.vit.get_intermediate_layers(x, n=self.vit.get_num_layers())
        hiddens_feature_maps = []
        for hidden in hiddens:

            patch_tokens = hidden[:, self.vit.n_cls_tokens :, :]
            npatch = int(patch_tokens.shape[1] ** 0.5)
            feature_map = einops.rearrange(
                patch_tokens,
                "b (npatch1 npatch2) c -> b npatch1 npatch2 c",
                npatch1=npatch,
                npatch2=npatch,
            )
            hiddens_feature_maps.append(feature_map)
        return hiddens_feature_maps


class SAMWrapperForUNETR(torch.nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        if hasattr(self.image_encoder, "neck"):
            del self.image_encoder.neck

    def forward(self, x):
        image_encoder = self.image_encoder
        x = image_encoder.patch_embed(x)
        if image_encoder.pos_embed is not None:
            x = x + interpolate_pos_encoding(x, image_encoder.pos_embed)

        hiddens = []
        for blk in image_encoder.blocks:
            x = blk(x)
            hiddens.append(x)

        return hiddens


class DinoV2WrapperForUNETR(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        n_blocks = len(self.vit.blocks)
        hiddens = self.vit.get_intermediate_layers(
            x, n=n_blocks, reshape=True
        )  # [B, C, H, W]
        hiddens_feature_maps = []
        for hidden in hiddens:
            hiddens_feature_maps.append(hidden.permute(0, 2, 3, 1))

        return hiddens_feature_maps


def _get_wrapped_image_encoder(image_encoder):
    if isinstance(image_encoder, VisionTransformer):
        encoder = VITImageEncoderWrapperForUNETR(image_encoder)
        encoder.embedding_size = image_encoder.embed_dim
        return encoder
    elif isinstance(image_encoder, SAM):
        return SAMWrapperForUNETR(image_encoder)
    elif isinstance(image_encoder, DinoV2WrapperForUNETR):
        return DinoV2WrapperForUNETR(image_encoder)
    else:
        raise ValueError(f"Unknown image encoder {image_encoder}")


@register_model 
def unetr_generic(*, backbone_cfg: dict, **kwargs): 
    image_encoder = create_model(**backbone_cfg)
    unetr = UNETR(
        image_encoder=image_encoder,
        **kwargs,
    )
    return unetr


@register_model
def unetr_vit(backbone="vit_small", backbone_kwargs={}, **kwargs):
    image_encoder = create_model(backbone, **backbone_kwargs)
    pretrained_cfg = image_encoder.__dict__.get("pretrained_cfg", None)
    image_encoder = _get_wrapped_image_encoder(image_encoder)
    unetr = UNETR(
        image_encoder=image_encoder,
        embedding_size=image_encoder.embedding_size,
        input_size=224,
        output_size=224,
        **kwargs,
    )
    if pretrained_cfg:
        unetr.pretrained_cfg = dict(
            mean=pretrained_cfg.mean,
            std=pretrained_cfg.std,
        )

    return unetr


@register_model
def sam_unetr(): 
    from .sam import build_sam

    sam = build_sam()
    image_encoder = SAMWrapperForUNETR(sam.image_encoder)
    model = UNETR(
        image_encoder,
        embedding_size=768,
        feature_size=64,
        input_size=224,
        output_size=224,
    )
    return model


@register_model
def medsam_unetr():
    from .medsam import medsam_vit_b

    sam = medsam_vit_b()
    image_encoder = SAMWrapperForUNETR(sam.image_encoder)
    model = UNETR(
        image_encoder,
        embedding_size=768,
        feature_size=64,
        input_size=224,
        output_size=224,
    )
    return model


@register_model
def medsam_unetr_prost_seg(
    feature_size=32, img_size=224, output_size=224, pretrained=True
):
    from medAI.modeling.sam import medsam

    backbone = medsam()
    backbone.image_encoder.embed_dim = 768
    image_encoder = SAMWrapperForUNETR(backbone.image_encoder)
    embed_dim = backbone.image_encoder.embed_dim

    unetr_model = UNETR(
        image_encoder,
        embedding_size=embed_dim,
        feature_size=feature_size,
        input_size=img_size,
        output_size=output_size,
        out_channels=2,
        norm_name="instance",
    )

    path = "/h/pwilson/projects/medAI/projects/medibot/medsam_unetr_best_v2.pt"
    if pretrained:
        sd = torch.load(path, map_location="cpu")
        unetr_model.load_state_dict(sd)

        from torchvision.transforms import v2 as T

        preprocess = T.Compose(
            [
                T.ToPILImage("RGB"),
                T.Resize((224, 224)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0, 0, 0], [1, 1, 1]),
            ]
        )

        unetr_model.preprocess = preprocess

    return unetr_model


@register_model
def usfm_unetr_in224(image_size=224):
    from .usfm import usfm_backbone

    model = usfm_backbone(image_size=image_size, out_indices=list(range(12)))

    class USFMWrapperForUNETR(nn.Module):
        def __init__(self, usfm):
            super().__init__()
            self.usfm = usfm

        def forward(self, x):
            outputs = self.usfm(x)
            outputs = [x.permute(0, 2, 3, 1) for x in outputs]
            return outputs

    model = USFMWrapperForUNETR(model)
    model = UNETR(
        model,
        embedding_size=768,
        feature_size=64,
        input_size=image_size,
        output_size=image_size,
    )
    return model


@register_model 
def dinov3_unetr(image_size=224, name='dinov3_vitl16', freeze_backbone=False, **kwargs): 
    import dinov3.hub.backbones
    backbone = dinov3.hub.backbones.__dict__[name](pretrained=True)
    embed_dim = backbone.embed_dim
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    backbone = DinoV2WrapperForUNETR(backbone)
    model = UNETR(
        backbone, embedding_size=embed_dim, input_size=image_size, **kwargs
    )
    return model


@register_model 
def dinov3_unetr_custom(*, image_size=224, backbone_cfg, freeze_backbone=False, **kwargs): 
    from medAI.modeling import create_model
    model = create_model(**backbone_cfg)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    backbone = model 
    embed_dim = backbone.embed_dim
    backbone = DinoV2WrapperForUNETR(backbone)

    model = UNETR(
        backbone,
        embedding_size=embed_dim,
        input_size=image_size,
        **kwargs
    )

    return model


if __name__ == "__main__":
    from .vision_transformer import vit_small

    vit = vit_small(patch_size=16, img_size=224)
    image_encoder = VITImageEncoderWrapperForUNETR(vit)
    model = UNETR(
        image_encoder,
        embedding_size=384,
        feature_size=14,
        input_size=256,
        output_size=64,
    )
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)

    model = medsam_unetr()
    x = torch.randn(1, 3, 1024, 1024)
    out = model(x)
    print(out.shape)