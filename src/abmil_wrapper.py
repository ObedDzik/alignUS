import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect

# class NeedleABMILWrapper(nn.Module):
#     def __init__(
#         self,
#         encoder: nn.Module,
#         abmil: nn.Module,
#         needle: np.ndarray,
#     ):
#         """
#         Args:
#             encoder:  DINOv2 (or similar) backbone. Its forward() should return
#                       a spatial feature map of shape (B, C, H, W).
#             abmil:    An ABMILISUP instance.
#             needle:   2-D boolean/uint8 mask of shape (img_H, img_W), e.g. (1024, 1024).
#                       Pixels that belong to the needle region should be True / 1.
#         """
#         super().__init__()
#         self.encoder = encoder
#         self.abmil = abmil
#         # Register as a buffer so it moves with .to(device) but is not a parameter
#         needle_tensor = torch.as_tensor(needle, dtype=torch.bool)
#         self.register_buffer("needle", needle_tensor)

    # def forward(self, bmode: torch.Tensor, return_h: bool = False):
    #     """
    #     Args:
    #         bmode:    Input image batch, shape (B, C, img_H, img_W).
    #         return_h: If True, also return the pre-logit hidden features from ABMIL.

    #     Returns (all tensors on the same device as bmode):
    #         logits:  (B, num_classes)
    #         feats:   (B, N, proj_dim)  — bag embedding after ABMIL projection
    #                  (N = number of needle tokens)
    #         A:       (B, N)            — attention weights over needle tokens
    #         h:       (B, cls_hidden)   — only present when return_h=True
    #     """
    #     outputs = self.encoder.forward_features(bmode)
    #     image_tokens = outputs['x_norm_patchtokens']  # (B, num_patches, C)
    #     B, num_patches, C = image_tokens.shape

    #     # Reconstruct spatial grid
    #     # DINOv2 patch_size=14 by default, so for 512x512 input: 512//14 = 36 patches per side
    #     patch_size = self.encoder.patch_size
    #     img_H, img_W = bmode.shape[-2], bmode.shape[-1]  # use actual input dims, not needle dims
    #     grid_H = img_H // patch_size
    #     grid_W = img_W // patch_size

    #     # Reshape to spatial: (B, grid_H, grid_W, C)
    #     image_tokens = image_tokens.view(B, grid_H, grid_W, C)

    #     # Downsample needle mask to feature-map resolution
    #     needle_H, needle_W = self.needle.shape
    #     patch_h = needle_H // grid_H
    #     patch_w = needle_W // grid_W

    #     needle_batch = self.needle.unsqueeze(0).expand(B, -1, -1)  # (B, needle_H, needle_W)

    #     mask_reshaped = needle_batch.view(B, grid_H, patch_h, grid_W, patch_w)
    #     token_mask = mask_reshaped.any(dim=(2, 4))  # (B, grid_H, grid_W)

    #     # Rest of the indexing stays the same...
    #     indices = torch.nonzero(token_mask[0], as_tuple=False)  # (N, 2)
    #     row_idx = indices[:, 0]
    #     col_idx = indices[:, 1]
    #     N = row_idx.shape[0]

    #     batch_idx = torch.arange(B, device=image_tokens.device).unsqueeze(1).expand(B, N)

    #     feats = image_tokens[
    #         batch_idx,
    #         row_idx.unsqueeze(0).expand(B, -1),
    #         col_idx.unsqueeze(0).expand(B, -1),
    #     ]  # (B, N, C)

    #     abmil_mask = token_mask[
    #         torch.arange(B, device=image_tokens.device).unsqueeze(1).expand(B, N),
    #         row_idx.unsqueeze(0).expand(B, -1),
    #         col_idx.unsqueeze(0).expand(B, -1),
    #     ]

    #     if return_h:
    #         logits, A, z, h = self.abmil(feats, mask=abmil_mask, return_h=True)
    #         return logits, A, z, h

    #     logits, A, z, H = self.abmil(feats, mask=abmil_mask, return_h=False)
    #     return logits, A, z, H


    # def forward(self, bmode: torch.Tensor, return_h: bool = False):
    #     outputs      = self.encoder.forward_features(bmode)
    #     image_tokens = outputs['x_norm_patchtokens']
    #     B, num_patches, C = image_tokens.shape

    #     patch_size = self.encoder.patch_size
    #     img_H, img_W = bmode.shape[-2], bmode.shape[-1]
    #     grid_H = img_H // patch_size
    #     grid_W = img_W // patch_size

    #     image_tokens = image_tokens.view(B, grid_H, grid_W, C)

    #     needle_H, needle_W = self.needle.shape
    #     patch_h = needle_H // grid_H
    #     patch_w = needle_W // grid_W
    #     needle_batch  = self.needle.unsqueeze(0).expand(B, -1, -1)
    #     mask_reshaped = needle_batch.view(B, grid_H, patch_h, grid_W, patch_w)
    #     token_mask    = mask_reshaped.any(dim=(2, 4))

    #     indices = torch.nonzero(token_mask[0], as_tuple=False)  # (N, 2)
    #     row_idx = indices[:, 0]
    #     col_idx = indices[:, 1]
    #     N       = row_idx.shape[0]

    #     batch_idx = torch.arange(B, device=image_tokens.device).unsqueeze(1).expand(B, N)
    #     feats = image_tokens[
    #         batch_idx,
    #         row_idx.unsqueeze(0).expand(B, -1),
    #         col_idx.unsqueeze(0).expand(B, -1),
    #     ]  # (B, N, C)

    #     abmil_mask = token_mask[
    #         torch.arange(B, device=image_tokens.device).unsqueeze(1).expand(B, N),
    #         row_idx.unsqueeze(0).expand(B, -1),
    #         col_idx.unsqueeze(0).expand(B, -1),
    #     ]

    #     logits, A, z, H = self.abmil(feats, mask=abmil_mask, return_h=False)

    #     return logits, A, z, H, indices, grid_H, grid_W


# class NeedleABMILWrapper(nn.Module):
#     def __init__(
#         self,
#         encoder: nn.Module,
#         abmil:   nn.Module,
#         model_type,
#         pooling: str = "attention",   # configurable, no longer hardcoded
#     ):
#         super().__init__()
#         self.encoder = encoder
#         self.abmil   = abmil
#         self.model_type = model_type
#         self.pooling = pooling

#     def forward(
#         self,
#         bmode:       torch.Tensor,   # (B, C, H, W)
#         needle_mask: torch.Tensor,   # (B, H, W) bool, from data batch
#     ):
#         device       = bmode.device
#         outputs      = self.encoder.forward_features(bmode)
#         image_tokens = outputs['x_norm_patchtokens']
#         B, num_patches, C = image_tokens.shape

#         patch_size   = self.encoder.patch_size
#         img_H, img_W = bmode.shape[-2], bmode.shape[-1]
#         grid_H       = img_H // patch_size
#         grid_W       = img_W // patch_size

#         image_tokens = image_tokens.view(B, grid_H, grid_W, C)

#         # Per-sample needle mask from batch
#         needle_batch  = needle_mask.to(device).bool()   # (B, H, W)
#         needle_H      = needle_batch.shape[-2]
#         needle_W      = needle_batch.shape[-1]
#         patch_h       = needle_H // grid_H
#         patch_w       = needle_W // grid_W

#         mask_reshaped = needle_batch.view(B, grid_H, patch_h, grid_W, patch_w)
#         token_mask    = mask_reshaped.any(dim=(2, 4))   # (B, grid_H, grid_W)

#         # Use first sample's token mask for shared needle indices
#         indices = torch.nonzero(token_mask[0], as_tuple=False)  # (N, 2)
#         row_idx = indices[:, 0]
#         col_idx = indices[:, 1]
#         N       = row_idx.shape[0]

#         batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

#         feats = image_tokens[
#             batch_idx,
#             row_idx.unsqueeze(0).expand(B, -1),
#             col_idx.unsqueeze(0).expand(B, -1),
#         ]  # (B, N, C)

#         abmil_mask = token_mask[
#             torch.arange(B, device=device).unsqueeze(1).expand(B, N),
#             row_idx.unsqueeze(0).expand(B, -1),
#             col_idx.unsqueeze(0).expand(B, -1),
#         ]

#         logits, A, z, H = self.abmil(feats, mask=abmil_mask, return_h=False, pooling=self.pooling)

#         return logits, A, z, H, indices, grid_H, grid_W

    



class NeedleABMILWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        abmil:   nn.Module,
        model_type,
        mask_size,
        pooling: str = "attention",  # configurable, no longer hardcoded
    ):
        super().__init__()
        self.encoder = encoder
        self.abmil   = abmil
        self.model_type = model_type
        self.pooling = pooling
        self.mask_size = mask_size

    def forward(
        self,
        bmode:       torch.Tensor,   # (B, C, H, W)
        needle_mask: torch.Tensor,   # (B, H, W) bool, from data batch
    ):
        device = bmode.device

        if self.model_type == 'dino':
            outputs      = self.encoder.forward_features(bmode)
            image_tokens = outputs['x_norm_patchtokens']

            B, num_patches, C = image_tokens.shape
            patch_size   = self.encoder.patch_size
            img_H, img_W = bmode.shape[-2], bmode.shape[-1]
            grid_H       = img_H // patch_size
            grid_W       = img_W // patch_size
            image_tokens = image_tokens.view(B, grid_H, grid_W, C)

        elif self.model_type == 'medsam':
            # for cls in type(self.encoder.image_encoder).mro():
            #     if "forward" in cls.__dict__:
            #         print("Forward found in:", cls)
            #         print("File:", inspect.getfile(cls.__dict__["forward"]))
            #         break
            image_tokens = self.encoder.image_encoder(bmode)
            image_tokens = image_tokens.permute(0, 2, 3, 1)
            B, grid_H, grid_W, C = image_tokens.shape

        elif self.model_type == 'microsegnet':
            image_tokens = self.encoder(bmode)              # (B, n_patch, C)
            B, n_patch, C = image_tokens.shape
            grid_H = grid_W = int(n_patch ** 0.5)
            if grid_H * grid_W != n_patch:
                raise ValueError(f"n_patch={n_patch} is not a perfect square.")
            image_tokens = image_tokens.view(B, grid_H, grid_W, C)
            img_H = img_W = self.encoder.img_size

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        needle_batch = needle_mask.to(device).bool()   # (B, mask_size, mask_size)
        needle_H, needle_W = needle_batch.shape[-2], needle_batch.shape[-1]

        expected_size = self.mask_size
        if needle_H != expected_size or needle_W != expected_size:
            raise ValueError(
                f"needle_mask shape ({needle_H}, {needle_W}) does not match "
                f"mask_size={expected_size}. Resize the mask in the "
                f"dataloader/transform, not here."
            )

        if needle_H % grid_H != 0 or needle_W % grid_W != 0:
            raise ValueError(
                f"needle_mask size ({needle_H}, {needle_W}) does not evenly "
                f"divide encoder grid ({grid_H}, {grid_W}) for "
                f"model_type='{self.model_type}'. Check mask_size "
                f"against the encoder's actual output resolution."
            )

        patch_h = needle_H // grid_H
        patch_w = needle_W // grid_W
        mask_reshaped = needle_batch.view(B, grid_H, patch_h, grid_W, patch_w)
        token_mask    = mask_reshaped.any(dim=(2, 4))   # (B, grid_H, grid_W)

        # Compute per-sample needle indices and pad to max N in batch
        indices_per_sample = [
            torch.nonzero(token_mask[b], as_tuple=False) for b in range(B)
        ]
        N_per_sample = [idx.shape[0] for idx in indices_per_sample]
        N_max = max(N_per_sample)

        # Build padded feats (B, N_max, C) and abmil_mask (B, N_max)
        C = image_tokens.shape[-1]
        feats    = torch.zeros(B, N_max, C, device=device)
        abmil_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        for b in range(B):
            idx = indices_per_sample[b]        # (N_b, 2)
            N_b = idx.shape[0]
            if N_b == 0:
                continue
            row_idx = idx[:, 0]
            col_idx = idx[:, 1]
            feats[b, :N_b] = image_tokens[b, row_idx, col_idx]   # (N_b, C)
            abmil_mask[b, :N_b] = True

        logits, A, z, H = self.abmil(
            feats, mask=abmil_mask, return_h=False, pooling=self.pooling
        )

        # Return first sample's indices for diagnostics/visualization only —
        # clearly labeled as sample-0-specific, not shared across the batch
        indices = indices_per_sample[0]

        # return logits, A, z, H, indices, grid_H, grid_W
        return logits, A, z, H, abmil_mask, indices, grid_H, grid_W







        # patch_h = needle_H // grid_H
        # patch_w = needle_W // grid_W
        # mask_reshaped = needle_batch.view(B, grid_H, patch_h, grid_W, patch_w)
        # token_mask    = mask_reshaped.any(dim=(2, 4))   # (B, grid_H, grid_W)

        # # Use first sample's token mask for shared needle indices
        # indices = torch.nonzero(token_mask[0], as_tuple=False)  # (N, 2)
        # row_idx = indices[:, 0]
        # col_idx = indices[:, 1]
        # N       = row_idx.shape[0]

        # batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

        # feats = image_tokens[
        #     batch_idx,
        #     row_idx.unsqueeze(0).expand(B, -1),
        #     col_idx.unsqueeze(0).expand(B, -1),
        # ]  # (B, N, C)

        # abmil_mask = token_mask[
        #     torch.arange(B, device=device).unsqueeze(1).expand(B, N),
        #     row_idx.unsqueeze(0).expand(B, -1),
        #     col_idx.unsqueeze(0).expand(B, -1),
        # ]

        # logits, A, z, H = self.abmil(
        #     feats, mask=abmil_mask, return_h=False, pooling=self.pooling
        # )

        # return logits, A, z, H, indices, grid_H, grid_W