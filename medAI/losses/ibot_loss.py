from torch import nn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np


class iBOTLoss(nn.Module):
    """
    iBOT loss function.

    Args:
        out_dim: The output dimension of the model.
        patch_out_dim: The output dimension of the patch model.
        ngcrops: The number of global crops.
        nlcrops: The number of local crops.
        warmup_teacher_temp: The warmup teacher temperature.
        teacher_temp: The teacher temperature.
        warmup_teacher_temp2: The warmup teacher temperature for the patch model.
        teacher_temp2: The teacher temperature for the patch model.
        warmup_teacher_temp_epochs: The number of warmup epochs for the teacher temperature.
        nepochs: The number of epochs.
        student_temp: The student temperature.
        center_momentum: The center momentum.
        center_momentum2: The center momentum for the patch model.
        lambda1: The lambda for the cls loss.
        lambda2: The lambda for the patch loss.
        mim_start_epoch: The epoch to start the mim loss.
    """

    def __init__(
        self,
        out_dim,
        patch_out_dim,
        ngcrops,
        nlcrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp2,
        teacher_temp2,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        mim_start_epoch=0,
    ):

        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.teacher_temp2_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_temp2,
                    np.linspace(
                        warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch)
                    * teacher_temp2,
                )
            )
        )

    def forward(
        self, student_output, teacher_output, student_local_cls, student_mask, epoch
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            # student cls is global crops + local crops
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(
                        dim=-1
                    ).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(
                        -teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1),
                        dim=-1,
                    )
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(
            cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2
        )
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2
<<<<<<<< HEAD:medAI/modeling/ibot/loss.py
        )


# class DINOLoss(nn.Module):
#     def __init__(
#         self,
#         out_dim,
#         ncrops,
#         warmup_teacher_temp,
#         teacher_temp,
#         warmup_teacher_temp_epochs,
#         nepochs,
#         student_temp=0.1,
#         center_momentum=0.9,
#     ):
#         super().__init__()
#         self.student_temp = student_temp
#         self.center_momentum = center_momentum
#         self.ncrops = ncrops
#         self.register_buffer("center", torch.zeros(1, out_dim))
#         # we apply a warm up for the teacher temperature because
#         # a too high temperature makes the training instable at the beginning
#         self.teacher_temp_schedule = np.concatenate(
#             (
#                 np.linspace(
#                     warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
#                 ),
#                 np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
#             )
#         )

#     def forward(self, student_output, teacher_output, epoch):
#         """
#         Cross-entropy between softmax outputs of the teacher and student networks.
#         """
#         student_out = student_output / self.student_temp
#         student_out = student_out.chunk(self.ncrops)
#         # teacher centering and sharpening
#         temp = self.teacher_temp_schedule[epoch]
#         teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
#         teacher_out = teacher_out.detach().chunk(2)

#         total_loss = 0
#         n_loss_terms = 0
#         for iq, q in enumerate(teacher_out):
#             for v in range(len(student_out)):
#                 if v == iq:
#                     # we skip cases where student and teacher operate on the same view
#                     continue
#                 loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
#                 total_loss += loss.mean()
#                 n_loss_terms += 1
#         total_loss /= n_loss_terms
#         self.update_center(teacher_output)
#         return total_loss

#     @torch.no_grad()
#     def update_center(self, teacher_output):
#         """
#         Update center used for teacher output.
#         """
#         batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
#         dist.all_reduce(batch_center)
#         batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

#         # ema update
#         self.center = self.center * self.center_momentum + batch_center * (
#             1 - self.center_momentum
#         )

class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        nglobal_crops=2,  # Add this parameter
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.nglobal_crops = nglobal_crops  # Store number of global crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Teacher temperature schedule
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nglobal_crops)  # FIXED: use nglobal_crops

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

class JEPALoss(nn.Module):
    def __init__(
        self,
        warmup_epochs=10,
        max_weight=1.0,
        loss_type='mse',  # 'mse', 'cosine', 'huber'
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
        self.loss_type = loss_type
    
    def forward(self, student_output, teacher_output, epoch):
        """
        Compute JEPA loss with epoch-dependent weighting.
        
        Args:
            student_output: [B, N, D] predictor embeddings
            teacher_output: [B, N, D] teacher embeddings (detached)
            epoch: current epoch for scheduling
        """
        # Compute base loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(student_output, teacher_output.detach())
        elif self.loss_type == 'cosine':
            # Cosine similarity loss (1 - cosine_sim)
            loss = 1 - F.cosine_similarity(
                student_output.reshape(-1, student_output.shape[-1]),
                teacher_output.detach().reshape(-1, teacher_output.shape[-1]),
                dim=-1
            ).mean()
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(student_output, teacher_output.detach())
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply warmup schedule
        weight = self._get_weight(epoch)
        
        return weight * loss
    
    def _get_weight(self, epoch):
        """Linear warmup for loss weight."""
        if epoch < self.warmup_epochs:
            return self.max_weight * (epoch / self.warmup_epochs)
        return self.max_weight

class ContrastiveJEPALoss(nn.Module):
    def __init__(self, temperature=0.1, warmup_epochs=10, max_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
    
    def forward(self, student_output, teacher_output, epoch):
        """
        Args:
            student_output: [B, N, D] predictor embeddings
            teacher_output: [B, N, D] CNN teacher embeddings
        """
        # Normalize embeddings
        student_norm = F.normalize(student_output, dim=-1)
        teacher_norm = F.normalize(teacher_output.detach(), dim=-1)
        
        # Flatten patches: [B, N, D] -> [B*N, D]
        B, N, D = student_norm.shape
        student_flat = student_norm.reshape(-1, D)
        teacher_flat = teacher_norm.reshape(-1, D)
        
        # Compute similarity matrix: [B*N, B*N]
        logits = torch.matmul(student_flat, teacher_flat.T) / self.temperature
        
        # Positive pairs are diagonal elements
        labels = torch.arange(B * N, device=logits.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(logits, labels)
        
        # Apply warmup
        weight = self._get_weight(epoch)
        return weight * loss
    
    def _get_weight(self, epoch):
        if epoch < self.warmup_epochs:
            return self.max_weight * (epoch / self.warmup_epochs)
        return self.max_weight
========
        )
>>>>>>>> origin/main:medAI/losses/ibot_loss.py
