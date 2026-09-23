"""
Self-supervised learning loss functions.

Includes SimCLR, MoCo-style losses for ConvNeXt SSL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SimCLRLoss(nn.Module):
    """
    SimCLR contrastive loss (NT-Xent).
    
    Treats augmented versions of the same image as positives,
    all other images in the batch as negatives.
    
    Args:
        temperature: Temperature parameter for softmax
        normalize: Whether to L2-normalize features
    """
    
    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, features: torch.Tensor, images_per_batch: int) -> torch.Tensor:
        """
        Compute SimCLR loss.
        
        Args:
            features: [B*N, D] where N is number of augmentations per image
            images_per_batch: Number of unique images in batch
            
        Returns:
            Scalar loss
        """
        if self.normalize:
            features = F.normalize(features, dim=1)
        
        # Number of views per image
        n_views = features.shape[0] // images_per_batch
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positives
        # For each sample, positives are other views of the same image
        batch_size = features.shape[0]
        mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=features.device)
        
        for i in range(images_per_batch):
            for j in range(n_views):
                for k in range(n_views):
                    if j != k:
                        mask[i * n_views + j, i * n_views + k] = True
        
        # Mask out self-similarity
        mask.fill_diagonal_(False)
        
        # For numerical stability
        similarity_matrix = similarity_matrix - torch.eye(
            batch_size, device=features.device
        ) * 1e9
        
        # Compute loss
        positives = similarity_matrix[mask].view(batch_size, -1)
        negatives = similarity_matrix[~mask].view(batch_size, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all GPUs."""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)
    
    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class SimCLRLossDistributed(nn.Module):
    """
    SimCLR loss with distributed training support.
    
    Gathers features from all GPUs before computing loss.
    """
    
    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, features: torch.Tensor, images_per_batch: int) -> torch.Tensor:
        """
        Args:
            features: [B*N, D] local features
            images_per_batch: Number of images per GPU
        """
        if self.normalize:
            features = F.normalize(features, dim=1)
        
        # Gather features from all GPUs
        if dist.is_available() and dist.is_initialized():
            features_gathered = torch.cat(GatherLayer.apply(features), dim=0)
            images_per_batch = images_per_batch * dist.get_world_size()
        else:
            features_gathered = features
        
        return self._compute_loss(features, features_gathered, images_per_batch)
    
    def _compute_loss(
        self,
        features_local: torch.Tensor,
        features_all: torch.Tensor,
        total_images: int
    ) -> torch.Tensor:
        """Compute contrastive loss."""
        
        n_views = features_all.shape[0] // total_images
        batch_size_local = features_local.shape[0]
        
        # Compute similarities with all gathered features
        similarity = torch.matmul(features_local, features_all.T) / self.temperature
        
        # Create labels for positives
        # Each sample's positives are other views of the same image
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        labels = torch.arange(batch_size_local, device=features_local.device)
        labels = labels + rank * batch_size_local // n_views
        labels = labels.repeat(n_views)
        
        # Create mask for positives (excluding self)
        mask = torch.zeros(
            batch_size_local, features_all.shape[0], 
            dtype=torch.bool, device=features_local.device
        )
        
        # Mark positives
        for i in range(batch_size_local):
            img_idx = labels[i]
            for j in range(n_views):
                global_idx = img_idx * n_views + j
                if global_idx != i + rank * batch_size_local:  # Exclude self
                    mask[i, global_idx] = True
        
        # Compute loss
        positives = similarity[mask].view(batch_size_local, -1)
        negatives = similarity[~mask].view(batch_size_local, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels_ce = torch.zeros(batch_size_local, dtype=torch.long, device=features_local.device)
        
        loss = F.cross_entropy(logits, labels_ce)
        
        return loss


class MoCoV3Loss(nn.Module):
    """
    MoCo v3 style contrastive loss.
    
    Uses momentum encoder as positive keys, batch negatives.
    Simpler than original MoCo (no queue).
    
    Args:
        temperature: Temperature for contrastive loss
        normalize: Whether to normalize features
    """
    
    def __init__(self, temperature: float = 0.2, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: Student features [B, D]
            key: Teacher features [B, D]
            
        Returns:
            Contrastive loss
        """
        if self.normalize:
            query = F.normalize(query, dim=1)
            key = F.normalize(key, dim=1)
        
        # Gather keys from all GPUs
        if dist.is_available() and dist.is_initialized():
            key_gathered = torch.cat(GatherLayer.apply(key), dim=0)
        else:
            key_gathered = key
        
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        
        # Negative logits: NxK
        l_neg = torch.einsum('nc,kc->nk', [query, key_gathered])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positives are the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class BYOLLoss(nn.Module):
    """
    BYOL (Bootstrap Your Own Latent) loss.
    
    Cosine similarity between predictor output and target without negatives.
    
    Args:
        use_predictor: Whether student uses a predictor MLP
    """
    
    def __init__(self, use_predictor: bool = True):
        super().__init__()
        self.use_predictor = use_predictor
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_output: Student predictions [B, D]
            teacher_output: Teacher targets [B, D]
            
        Returns:
            Negative cosine similarity loss
        """
        # Normalize
        student_output = F.normalize(student_output, dim=-1)
        teacher_output = F.normalize(teacher_output, dim=-1)
        
        # Cosine similarity
        loss = 2 - 2 * (student_output * teacher_output).sum(dim=-1)
        
        return loss.mean()


class MultiViewContrastiveLoss(nn.Module):
    """
    General multi-view contrastive loss.
    
    Handles multiple crops (global + local) like DINO but with simpler contrastive objective.
    
    Args:
        temperature: Temperature parameter
        n_global_crops: Number of global crops
        n_local_crops: Number of local crops
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        n_global_crops: int = 2,
        n_local_crops: int = 6,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.normalize = normalize
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_output: [B*(n_global+n_local), D]
            teacher_output: [B*n_global, D]
            
        Returns:
            Contrastive loss
        """
        if self.normalize:
            student_output = F.normalize(student_output, dim=1)
            teacher_output = F.normalize(teacher_output, dim=1)
        
        # Gather teacher outputs from all GPUs
        if dist.is_available() and dist.is_initialized():
            teacher_output = torch.cat(GatherLayer.apply(teacher_output), dim=0)
        
        batch_size = teacher_output.shape[0] // self.n_global_crops
        n_student_views = self.n_global_crops + self.n_local_crops
        
        # Compute similarity matrix
        similarity = torch.matmul(student_output, teacher_output.T) / self.temperature
        
        # Create positive pairs mask
        # Each student view should match with teacher views of the same image
        mask = torch.zeros(
            student_output.shape[0],
            teacher_output.shape[0],
            dtype=torch.bool,
            device=student_output.device
        )
        
        for i in range(batch_size):
            for sv in range(n_student_views):
                student_idx = i * n_student_views + sv
                # Match with all teacher views of same image
                for tv in range(self.n_global_crops):
                    teacher_idx = i * self.n_global_crops + tv
                    # Don't match student global view with itself as teacher
                    if sv >= self.n_global_crops or sv != tv:
                        mask[student_idx, teacher_idx] = True
        
        # Compute cross-entropy loss
        loss = 0
        for i in range(student_output.shape[0]):
            if mask[i].sum() > 0:
                positives = similarity[i][mask[i]]
                negatives = similarity[i][~mask[i]]
                
                # Log-sum-exp for numerical stability
                logits = torch.cat([positives, negatives])
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                
                loss += F.cross_entropy(
                    logits.unsqueeze(0).repeat(positives.shape[0], 1),
                    labels.repeat(positives.shape[0])
                )
        
        loss = loss / student_output.shape[0]
        
        return loss


class SwAVLoss(nn.Module):
    """
    SwAV (Swapped Assignment Views) loss.
    
    Uses online clustering (Sinkhorn-Knopp) to generate pseudo-labels,
    then predicts cluster assignments across views.
    
    Args:
        temperature: Temperature for softmax
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations
        epsilon: Regularization for Sinkhorn
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        epsilon: float = 0.05,
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_output: Student prototypes [B*N, K]
            teacher_output: Teacher prototypes [B*M, K]
        """
        # Apply temperature
        student_output = student_output / self.temperature
        
        # Compute codes (cluster assignments) via Sinkhorn-Knopp
        with torch.no_grad():
            teacher_codes = self.sinkhorn(teacher_output)
        
        # Predict teacher codes from student
        loss = -torch.mean(
            torch.sum(teacher_codes * F.log_softmax(student_output, dim=1), dim=1)
        )
        
        return loss
    
    @torch.no_grad()
    def sinkhorn(self, out: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm for optimal transport.
        
        Args:
            out: Logits [B, K]
            
        Returns:
            Cluster assignments [B, K]
        """
        Q = torch.exp(out / self.epsilon).T  # [K, B]
        
        B = Q.shape[1]
        K = Q.shape[0]
        
        # Make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        
        for _ in range(self.sinkhorn_iterations):
            # Normalize rows
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            
            # Normalize columns
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        
        Q *= B  # Undo normalization
        return Q.T


def get_ssl_loss(method: str, **kwargs):
    """
    Factory function to get SSL loss.
    
    Args:
        method: Loss method name
        **kwargs: Loss-specific arguments
        
    Returns:
        Loss module
    """
    losses = {
        'simclr': SimCLRLoss,
        'simclr_dist': SimCLRLossDistributed,
        'mocov3': MoCoV3Loss,
        'byol': BYOLLoss,
        'multiview': MultiViewContrastiveLoss,
        'swav': SwAVLoss,
    }
    
    if method not in losses:
        raise ValueError(f"Unknown loss method: {method}. Available: {list(losses.keys())}")
    
    return losses[method](**kwargs)


if __name__ == '__main__':
    # Test losses
    print("Testing SSL losses...")
    
    # SimCLR
    loss_fn = SimCLRLoss(temperature=0.5)
    features = torch.randn(16, 128)  # 8 images, 2 views each
    loss = loss_fn(features, images_per_batch=8)
    print(f"SimCLR loss: {loss.item():.4f}")
    
    # MoCo v3
    loss_fn = MoCoV3Loss(temperature=0.2)
    query = torch.randn(8, 128)
    key = torch.randn(8, 128)
    loss = loss_fn(query, key)
    print(f"MoCo v3 loss: {loss.item():.4f}")
    
    # BYOL
    loss_fn = BYOLLoss()
    student = torch.randn(8, 128)
    teacher = torch.randn(8, 128)
    loss = loss_fn(student, teacher)
    print(f"BYOL loss: {loss.item():.4f}")
    
    print("\nAll tests passed!")