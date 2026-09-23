import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy


class MaskedAutoEncode(nn.Module):
    """
    Masked Autoencoding module for test-time training on ultrasound images.
    Updates the ENCODER at test time using reconstruction as self-supervision.
    The decoder is just a auxiliary head for the self-supervised signal.
    """

    def __init__(self, model, cfg, embed_dim=1024, patch_size=16, decoder_depth=3):
        super().__init__()
        self.patch_size = patch_size
        self.model = model
        self.device = cfg.device
        self.embed_dim = embed_dim
        
        # Lightweight decoder - just for self-supervision signal
        decoder_layers = []
        dims = [embed_dim, 512, patch_size * patch_size * 3]
        
        for i in range(len(dims) - 1):
            decoder_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.GELU() if i < len(dims) - 2 else nn.Identity()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        
        # Store original encoder state for optional reset
        self.original_encoder_state = None
    
    def save_original_state(self):
        """Save the original encoder state before test-time adaptation."""
        self.original_encoder_state = copy.deepcopy(
            self.model.model.image_encoder.state_dict()
        )
    
    def restore_original_state(self):
        """Restore encoder to original state (before adaptation)."""
        if self.original_encoder_state is not None:
            self.model.model.image_encoder.load_state_dict(self.original_encoder_state)
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform random masking by per-sample shuffling.
        
        Args:
            x: [B, N, D] input tokens
            mask_ratio: fraction of patches to mask
            
        Returns:
            mask: binary mask (1 = masked, 0 = kept)
            ids_restore: indices to restore original order
        """
        B, N, D = x.shape
        num_masked = int(N * mask_ratio)
        
        # Generate random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Create binary mask: 1 is masked, 0 is kept
        mask = torch.zeros(B, N, device=x.device)
        mask[:, :num_masked] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return mask, ids_restore
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            imgs: [B, C, H, W]
            
        Returns:
            patches: [B, N, patch_size^2 * C]
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        
        assert H % p == 0 and W % p == 0, f"Image size ({H}x{W}) must be divisible by patch size ({p})"
        
        h = H // p
        w = W // p
        
        # Reshape to patches
        patches = imgs.reshape(B, C, h, p, w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1)  # [B, h, w, p, p, C]
        patches = patches.reshape(B, h * w, p * p * C)
        
        return patches
    
    def forward(self, data: Dict, mask_ratio: float = 0.5, 
                use_normalized_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masked autoencoding.
        
        Args:
            data: dictionary containing 'bmode' images [B, 3, H, W]
            mask_ratio: fraction of patches to mask (0.5-0.75 typical for TTT)
            use_normalized_loss: whether to normalize loss by patch variance
            
        Returns:
            dict with 'loss' and 'mask'
        """
        images = data['bmode'].to(self.device)
        B, C, H, W = images.shape
        
        # Extract patch tokens from backbone - WITH GRADIENTS for encoder update
        tokens = self.model.model.image_encoder.backbone.get_intermediate_layers(
            images, n=1, reshape=False
        )
        
        patch_features = tokens[0]  # [B, N, D]
        B, N, D = patch_features.shape
        
        # Apply masking
        mask, ids_restore = self.random_masking(patch_features, mask_ratio)
        
        # Only process masked patches
        mask_indices = mask.bool()
        masked_tokens = patch_features[mask_indices]  # [B*num_masked, D]
        
        # Decode masked patches
        pred_patches = self.decoder(masked_tokens)  # [B*num_masked, patch_size^2*C]
        
        # Get ground truth patches
        gt_patches = self.patchify(images)  # [B, N, patch_size^2*C]
        gt_masked = gt_patches[mask_indices]  # [B*num_masked, patch_size^2*C]
        
        # Compute loss
        if use_normalized_loss:
            # Normalize by patch variance (helps with varying contrast in ultrasound)
            var = gt_masked.var(dim=-1, keepdim=True)
            loss = ((pred_patches - gt_masked) ** 2 / (var + 1e-6)).mean()
        else:
            loss = F.mse_loss(pred_patches, gt_masked)
        
        return {
            'loss': loss,
            'mask': mask,
            'num_masked': mask.sum().item() / B
        }
    
    def test_time_adapt(self, data: Dict, num_steps: int = 10, 
                        lr: float = 1e-4, mask_ratio: float = 0.6,
                        update_encoder_only: bool = True,
                        which_blocks: str = 'last_n',
                        n_blocks: int = 4) -> float:
        """
        Perform test-time adaptation by updating the ENCODER using masked autoencoding.
        
        Args:
            data: input data dictionary
            num_steps: number of TTT optimization steps (10-20 typical)
            lr: learning rate for adaptation (1e-4 to 1e-3)
            mask_ratio: masking ratio for reconstruction (0.5-0.75)
            update_encoder_only: if True, only update encoder. If False, update both.
            which_blocks: 'all', 'last_n', or 'last_only' - which encoder blocks to update
            n_blocks: if which_blocks='last_n', how many blocks to update
            
        Returns:
            final reconstruction loss
        """
        # Set encoder to training mode for BatchNorm/Dropout
        self.model.model.image_encoder.train()
        
        # Determine which parameters to update
        if update_encoder_only:
            # Only update encoder parameters
            if which_blocks == 'all':
                params_to_update = self.model.model.image_encoder.parameters()
            elif which_blocks == 'last_n':
                # Update only last n transformer blocks
                backbone = self.model.model.image_encoder.backbone
                if hasattr(backbone, 'blocks'):
                    params_to_update = list(backbone.blocks[-n_blocks:].parameters())
                else:
                    # Fallback to all encoder params
                    params_to_update = self.model.model.image_encoder.parameters()
            elif which_blocks == 'last_only':
                # Update only the very last block
                backbone = self.model.model.image_encoder.backbone
                if hasattr(backbone, 'blocks'):
                    params_to_update = list(backbone.blocks[-1].parameters())
                else:
                    params_to_update = self.model.model.image_encoder.parameters()
        else:
            # Update both encoder and decoder
            params_to_update = list(self.model.model.image_encoder.parameters()) + \
                             list(self.decoder.parameters())
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(params_to_update, lr=lr, weight_decay=0.01)
        
        final_loss = 0.0
        for step in range(num_steps):
            optimizer.zero_grad()
            
            output = self.forward(data, mask_ratio=mask_ratio)
            loss = output['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_update, 1.0)
            optimizer.step()
            
            final_loss = loss.item()
        
        # Set back to eval mode for inference
        self.model.model.image_encoder.eval()
        
        return final_loss
    
    def adapt_and_predict(self, data: Dict, num_steps: int = 10, 
                          lr: float = 1e-4, mask_ratio: float = 0.6,
                          reset_after: bool = False) -> torch.Tensor:
        """
        Convenience method: adapt encoder to test sample, then make prediction.
        
        Args:
            data: input data dictionary
            num_steps: TTT steps
            lr: learning rate
            mask_ratio: masking ratio
            reset_after: if True, restore encoder to original state after prediction
            
        Returns:
            predictions from the adapted model
        """
        # Adapt encoder to this test sample
        self.test_time_adapt(data, num_steps=num_steps, lr=lr, mask_ratio=mask_ratio)
        
        # Make prediction with adapted encoder
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(data)
        
        # Optionally reset encoder
        if reset_after and self.original_encoder_state is not None:
            self.restore_original_state()
        
        return predictions