import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from baseline.guideus.src.baseline_attention_reg import ACMILHead

class ABMILISUP(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 6,
        proj_dim: int = 768,
        attn_hidden: int = 512,
        cls_hidden: int = 256,
        p_input_dropout: float = 0.10,
        p_attn_dropout: float = 0.15,
        p_cls_dropout: float = 0.15,
        use_acmil: bool = False,         
        acmil_n_branches: int = 5,
        acmil_n_masked_patch: int = 10,
        acmil_mask_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(p_input_dropout)
        self.proj = nn.Linear(input_dim, proj_dim)
        self.proj_bn = nn.LayerNorm(proj_dim)

        self.use_acmil = use_acmil
        if self.use_acmil:
            self.acmil_head = ACMILHead(
                proj_dim=proj_dim,
                attn_hidden=attn_hidden,
                n_branches=acmil_n_branches,
                n_masked_patch=acmil_n_masked_patch,
                mask_drop_prob=acmil_mask_drop_prob,
                p_attn_dropout=p_attn_dropout,
            )
        else:
            # Gated attention
            self.attn_V = nn.Linear(proj_dim, attn_hidden)
            self.attn_U = nn.Linear(proj_dim, attn_hidden)
            self.attn_w = nn.Linear(attn_hidden, 1)
            self.attn_do = nn.Dropout(p_attn_dropout)

        self.cls_fc1 = nn.Linear(proj_dim, cls_hidden)
        self.cls_bn1 = nn.LayerNorm(cls_hidden)
        self.cls_do1 = nn.Dropout(p_cls_dropout)
        self.cls_fc2 = nn.Linear(cls_hidden, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, mask=None, return_h: bool = False, pooling: str = "attention"):
        squeeze_back = False
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeeze_back = True

        B, N, _ = X.shape
        device = X.device

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        X = self.input_dropout(X)

        H = self.proj_bn(self.proj(X))
        H = self.relu(H)

        if self.use_acmil:
            z, A, A_branches = self.acmil_head(H, mask=mask)
        else:
            if pooling == "attention":
                Vh = torch.tanh(self.attn_V(H))
                Uh = torch.sigmoid(self.attn_U(H))
                gate = Vh * Uh
                gate = self.attn_do(gate)

                A = self.attn_w(gate).squeeze(-1)
                A = A.masked_fill(~mask, torch.finfo(A.dtype).min)
                A = F.softmax(A, dim=1)
                z = torch.bmm(A.unsqueeze(1), H).squeeze(1)

            elif pooling == "mean":
                # Masked mean over valid (needle) patches only
                mask_f = mask.unsqueeze(-1).float()          # (B, N, 1)
                H_masked = H * mask_f
                z = H_masked.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)  # (B, D)
                # Uniform "attention" for logging/diagnostic compatibility
                A = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1e-8)

            else:
                raise ValueError(f"Unknown pooling mode: {pooling}")

        h = self.relu(self.cls_bn1(self.cls_fc1(z)))
        h = self.cls_do1(h)
        logits = self.cls_fc2(h)

        if squeeze_back:
            logits = logits.squeeze(0)
            A = A.squeeze(0)
            z = z.squeeze(0)

        if return_h:
            return logits, A, z, h, H
        return logits, A, z, H