import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from src.transunet.vit_seg_modeling import VisionTransformer as ViT_seg

class WrapperMicroSegNet(nn.Module):
    """
    Encoder-only wrapper around MicroSegNet's hybrid ResNet50-ViT transformer,
    with the segmentation decoder discarded. Intended to be passed as the
    `encoder=` argument to NeedleABMILWrapper, matching the same pattern
    used for the DINOv3 and MedSAM encoders.

    Exposes a forward() that returns raw patch tokens (B, n_patch, hidden),
    analogous to DINOv3's forward_features()['x_norm_patchtokens'] and
    MedSAM's image_encoder() output, so NeedleABMILWrapper's model_type
    branching can call it uniformly.
    """

    def __init__(
        self,
        checkpoint_path: str,
        img_size: int = 224,
        n_skip: int = 3,
        vit_name: str = 'R50-ViT-B_16',
        vit_patches_size: int = 16,
        freeze: bool = False,
    ):
        super().__init__()
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = 1
        config_vit.n_skip = n_skip
        config_vit.patches.size = (vit_patches_size, vit_patches_size)
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size),
                int(img_size / vit_patches_size),
            )

        full_net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
        full_net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        # Keep only the hybrid CNN stem + ViT encoder; discard the
        # segmentation decoder/head entirely -- not needed for grading.
        self.transformer = full_net.transformer
        del full_net.decoder
        del full_net.segmentation_head
        del full_net  # partially dismantled; do not reuse

        self.img_size = img_size

    def forward(self, bmode: torch.Tensor):
        """
        Args
        ----
        bmode : (B, C, H, W), C in {1, 3}, any spatial size

        Returns
        -------
        patch_tokens : (B, n_patch, hidden)
        """
        x = bmode
        if x.max() > 1.0:
            x = x / 255.0
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False,
            )

        patch_tokens, attn_weights, features = self.transformer(x)
        return patch_tokens


