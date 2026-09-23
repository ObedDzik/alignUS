import torch
import torch.nn as nn
import torch.nn.functional as F
from medAI.modeling._medsam.segment_anything.modeling.transformer import (
    TwoWayTransformer,
)
from medAI.layers.common import LayerNorm2d
from ._medsam.segment_anything.modeling.mask_decoder import MaskDecoder, ClassDecoder, MLP
from ._medsam.segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom

from argparse import ArgumentParser, Action, _StoreAction, _StoreTrueAction, FileType
import os
import typing as tp
from warnings import warn
import torch
from torch import nn
import logging
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm


class Proside(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dino_embedding_dim: int = 1024,
        floating_point_prompts: list[str] = [],
        discrete_prompts: list[str] = [],
        discrete_prompt_nvals: list[int] = [],
        use_class_decoder: bool = True,
        # --- sensible defaults ---
        aux_decoder_dim: int = 512,
        transformer_depth: int = 2,
        transformer_mlp_dim: int = 2048,
        transformer_heads: int = 8,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        cls_output_dims: list[int] = [2],
        num_data_independent_prompts: int = 0,
        prompt_embedding_dim: int | None = None,
        prompt_dropout: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.prompt_dropout = prompt_dropout
        self.floating_point_prompts = floating_point_prompts
        assert len(discrete_prompts) == len(discrete_prompt_nvals), (
            f"discrete_prompts ({len(discrete_prompts)}) and "
            f"discrete_prompt_nvals ({len(discrete_prompt_nvals)}) must be the same length"
        )
        self.discrete_prompts = discrete_prompts
        self.model.return_dict = True

        # decoder_dim is the single source of truth for all aux decoder widths
        self.decoder_dim = aux_decoder_dim if aux_decoder_dim is not None else dino_embedding_dim
        # prompt_embedding_dim defaults to decoder_dim so prompts slot in without projection
        self.prompt_embedding_dim = prompt_embedding_dim if prompt_embedding_dim is not None else self.decoder_dim
        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))

        # ── optional projection (only built when dims differ) ──────────
        # If aux_decoder_dim is None, decoder_dim == dino_embedding_dim, so no proj needed.
        if dino_embedding_dim != self.decoder_dim:
            self.feature_proj = nn.Sequential(
                nn.Conv2d(dino_embedding_dim, self.decoder_dim, kernel_size=1, bias=False),
                LayerNorm2d(self.decoder_dim),
            )
        else:
            self.feature_proj = nn.Identity()

        # ── positional encoding ────────────────────────────────────────
        # Learned 2D grid at a nominal 64×64; interpolated at runtime to
        # match the actual DINOv3 feature map spatial size.
        # self.pe_layer = nn.Parameter(
        #     torch.zeros(1, self.decoder_dim, 64, 64)
        # )
        # nn.init.normal_(self.pe_layer, std=0.02)

        self.pe_layer = PositionEmbeddingRandom(self.decoder_dim // 2)

        # ── dense no-mask embedding ────────────────────────────────────
        self.no_mask_embed = nn.Embedding(1, self.decoder_dim)

        # ── transformer factory ────────────────────────────────────────
        # num_heads must divide decoder_dim evenly.
        # With decoder_dim=1024, 8 heads → 128 dim/head (fine).
        assert self.decoder_dim % transformer_heads == 0, (
            f"decoder_dim ({self.decoder_dim}) must be divisible by "
            f"transformer_heads ({transformer_heads})"
        )

        def _make_transformer():
            return TwoWayTransformer(
                depth=transformer_depth,
                embedding_dim=self.decoder_dim,
                mlp_dim=transformer_mlp_dim,
                num_heads=transformer_heads,
            )

        # ── MaskDecoder ────────────────────────────────────────────────
        self.mask_decoder = MaskDecoder(
            transformer_dim=self.decoder_dim,
            transformer=_make_transformer(),
            num_multimask_outputs=num_multimask_outputs,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            output_upscaling_version='interpolate',
        )

        # ── ClassDecoder ───────────────────────────────────────────────
        self.class_decoder = (
            ClassDecoder(
                transformer_dim=self.decoder_dim,
                transformer=_make_transformer(),
                num_cls_tokens=len(cls_output_dims),
                cls_output_dims=cls_output_dims,
            )
            if use_class_decoder
            else None
        )

        # ── prompt modules ─────────────────────────────────────────────
        # If prompt_embedding_dim != decoder_dim, we need a small projection
        # to align prompt tokens before concatenating into sparse embeddings.
        if self.prompt_embedding_dim != self.decoder_dim:
            self.prompt_proj = nn.Linear(self.prompt_embedding_dim, self.decoder_dim, bias=False)
        else:
            self.prompt_proj = nn.Identity()

        self.null_prompt = nn.Parameter(torch.zeros(1, self.prompt_embedding_dim))

        self.floating_point_prompt_modules = nn.ModuleDict({
            p: nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, self.prompt_embedding_dim),
            )
            for p in floating_point_prompts
        })
        self.integer_prompt_modules = nn.ModuleDict({
            p: nn.Embedding(n, self.prompt_embedding_dim)
            for p, n in zip(discrete_prompts, discrete_prompt_nvals)
        })
        if num_data_independent_prompts > 0:
            self.data_independent_prompts = nn.Parameter(
                torch.randn(1, num_data_independent_prompts, self.prompt_embedding_dim)
            )
        else:
            self.data_independent_prompts = None

    # ── helpers ────────────────────────────────────────────────────────

    def _build_sparse_embedding(
        self, B: int, device: torch.device, **prompts
    ) -> torch.Tensor:
        tokens = []

        for prompt_name, prompt_value in prompts.items():
            dropped = (
                prompt_value is None
                or (self.prompt_dropout > 0 and self.training and torch.rand(1) < self.prompt_dropout)
            )
            if dropped:
                emb = self.null_prompt.expand(B, -1)
            elif prompt_name in self.floating_point_prompts:
                emb = self.floating_point_prompt_modules[prompt_name](prompt_value)
            elif prompt_name in self.discrete_prompts:
                emb = self.integer_prompt_modules[prompt_name](prompt_value)
            else:
                raise ValueError(f"Unknown prompt: {prompt_name}")

            # project to decoder_dim if needed, then add sequence dim
            tokens.append(self.prompt_proj(emb).unsqueeze(1))  # B, 1, decoder_dim

        if self.data_independent_prompts is not None:
            tokens.append(
                self.prompt_proj(
                    self.data_independent_prompts.expand(B, -1, -1)
                )
            )

        if tokens:
            return torch.cat(tokens, dim=1)  # B, N_prompts, decoder_dim
        else:
            # decoders require at least one sparse token
            return self.prompt_proj(self.null_prompt).unsqueeze(0).expand(B, 1, -1)

    # def _get_dense_and_pe(self, image_feats):
    #     B, C, H, W = image_feats.shape
    #     dense = self.no_mask_embed.weight.reshape(1, C, 1, 1).expand(B, C, H, W)
    #     pe = F.interpolate(
    #         self.pe_layer, size=(H, W), mode="bilinear", align_corners=False
    #     )  # keep as 1, C, H, W — MaskDecoder expands it internally
    #     return dense, pe

    def _get_dense_and_pe(self, image_feats):
        B, C, H, W = image_feats.shape
        dense = self.no_mask_embed.weight.reshape(1, C, 1, 1).expand(B, C, H, W)
        pe = self.pe_layer((H, W)).unsqueeze(0)  # 1, decoder_dim, H, W
        # no interpolation needed — generates directly at any resolution
        return dense, pe

    # ── forward ────────────────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,
        output_mode: str = "all",
        **prompts,
    ):
        B, device = image.shape[0], image.device
        _, _, ht,wd = image.shape

        model_out = self.model(image)
        heatmap_logits = model_out["cancer_logits"]

        # use final_feats (post-norm) instead of image_feats (pre-norm last hidden)
        raw_feats = model_out["final_feats"]  # this is the forward_features() dict

        # extract patch tokens and reshape to spatial
        patch_tokens = raw_feats["x_norm_patchtokens"]          # B, H*W, C
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)                                   # 32×32 for 512 input, patch=16
        raw_feats = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W

        image_feats = self.feature_proj(raw_feats)              # B, decoder_dim, H, W

        # 3. Build embeddings
        sparse = self._build_sparse_embedding(B, device, **prompts)
        dense, pe = self._get_dense_and_pe(image_feats)

        # 4. Aux mask decoder
        mask_logits, iou_pred = self.mask_decoder(
            image_embeddings=image_feats,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        mask_logits = (
            mask_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )
        # mask_logits = F.interpolate(
        #     mask_logits, 
        #     size=(ht, wd),  # original input spatial size
        #     mode="bilinear", 
        #     align_corners=False
        # )

        # 5. Aux class decoder
        cls_outputs = None
        if self.class_decoder is not None:
            cls_outputs = self.class_decoder(
                image_embeddings=image_feats,
                image_pe=pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
            )

        if output_mode == "heatmaps":
            return heatmap_logits
        elif output_mode == "classifier":
            assert cls_outputs is not None
            return cls_outputs[0]
        else:
            return dict(
                heatmap_logits=heatmap_logits,
                mask_logits=mask_logits,
                iou_pred=iou_pred,
                cls_outputs=cls_outputs,
            )