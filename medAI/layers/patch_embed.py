from typing import Sequence
import torch 
from torch import nn 
from einops import pack 
from einops.layers.torch import Rearrange


def pair(obj):
    if isinstance(obj, Sequence):
        assert len(obj) == 2
        return obj
    else:
        return obj, obj



class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
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


class FFTPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
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


class FFTPatchEmbedV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

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
                c=in_chans
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim//2),
            nn.LayerNorm(embed_dim //2),
        )

        self.to_freq_patch = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> (b h w) c p1 p2", p1 = freq_patch_height, p2 = freq_patch_width),
        )

        self.to_freq_embedding = nn.Sequential(
            Rearrange("(b d) c p1 p2 ri -> b d (p1 p2 ri c)", p1 = freq_patch_height, p2 = freq_patch_width, d=patch_dims),
            nn.LayerNorm(freq_patch_dim),
            nn.Linear(freq_patch_dim, embed_dim//2),
            nn.LayerNorm(embed_dim // 2)
        )

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            Rearrange("b (h w) d -> b d h w", h=h, w=w)
        )

    def forward(self, x): 
        patch_emb = self.to_patch_embedding(x)
        patch_for_freq = self.to_freq_patch(x)
        freqs = torch.fft.fft2(patch_for_freq)
        freqs = torch.view_as_real(freqs)
        freqs_emb = self.to_freq_embedding(freqs)

        merged_embs, _ = pack([freqs_emb, patch_emb], 'b n *')
        
        out = self.to_out(merged_embs)
        return out 
