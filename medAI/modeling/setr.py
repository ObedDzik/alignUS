from functools import partial
from torch import nn
import torch
from .vision_transformer import VisionTransformer
import math
from .registry import register_model


def make_upsample_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
    )


class SegmentationHead(nn.Module): 
    def __init__(self, in_channels, n_classes=1, n_layers=3, norm_layer=nn.BatchNorm2d): 
        super().__init__()
        self.layers = nn.ModuleList([
            TransposeConvBlock(in_channels, in_channels, norm_layer=norm_layer) for _ in range(n_layers)
        ])
        self.final_conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x): 
        for layer in self.layers: 
            x = layer(x)
        x = self.final_conv(x)
        return x


class TransposeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv = make_upsample_conv(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class SETR(VisionTransformer):
    def __init__(self, *args, embed_dim=768, num_classes=1, upsample_layers=4, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(*args, embed_dim=embed_dim, num_classes=num_classes, **kwargs)
        self.head = None
        self.segmentation_head = SegmentationHead(embed_dim, num_classes, upsample_layers, norm_layer)

    def forward(self, x):
        transformer_outputs = super().forward(x, return_all_tokens=True) # B, N, C 
        B, N, C = transformer_outputs.shape
        H = W = int(math.sqrt(N - 1))
        
        patch_tokens = transformer_outputs[:, self.n_cls_tokens:, :] # B, N-1, C 
        patch_tokens = patch_tokens.transpose(1, 2) # B, C, N-1 
        patch_tokens = patch_tokens.reshape(B, C, H, W) # B, C, H, W 

        x = self.segmentation_head(patch_tokens)
        return x

    
@register_model
def setr_tiny(**kwargs):
    model_args = dict(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return SETR(**dict(model_args, **kwargs))


@register_model
def setr_small(**kwargs):
    model_args = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return SETR(**dict(model_args, **kwargs))


@register_model
def setr_base(**kwargs):  
    model_args = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )
    return SETR(**dict(model_args, **kwargs))


@register_model
def setr_large(**kwargs):
    model_args = dict(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,  
    )
    return SETR(**dict(model_args, **kwargs))


@register_model
def setr_small_deit(**kwargs):
    model_args = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        n_cls_tokens=2,
    )
    return SETR(**dict(model_args, **kwargs))


@register_model 
def setr_small_deit_up2_in(**kwargs):
    model_args = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        n_cls_tokens=2,
        upsample_layers=2,
        norm_layer = nn.InstanceNorm2d
    )
    return SETR(**dict(model_args, **kwargs))


if __name__ == "__main__":
    inp = torch.randn(1, 3, 224, 224)
    setr = SETR()
    out = setr(inp)
    print(out.shape)
