# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type


# =============
# PositionEmbeddingRandom is copied from:
# github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py
# with licence:
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
# =============


class BoundingBoxEmbedding(nn.Module): 
    def __init__(self, num_pos_feats, **kwargs): 
        super().__init__()
        self.position_embedding = PositionEmbeddingRandom(num_pos_feats // 2, **kwargs)

    def forward(self, boxes_xyxy, image_size): 
        """
        Args: 
            boxes_xyxy: (B, N, 4) in (x1, y1, x2, y2) format
            image_size: (height, width)
        """

        top_left = boxes_xyxy[:, :, :2]
        bottom_right = boxes_xyxy[:, :, 2:]
        top_left_emb = self.position_embedding.forward_with_coords(
            top_left, image_size=image_size
        )  # (B, N, C)

        bottom_right_emb = self.position_embedding.forward_with_coords(
            bottom_right, image_size=image_size
        )  # (B, N, C)

        return torch.cat([top_left_emb, bottom_right_emb], dim=-1)  # (B, N, 2C)
