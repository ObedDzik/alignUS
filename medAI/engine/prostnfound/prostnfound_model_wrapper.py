from medAI.layers.masked_prediction_module import MaskedPredictionModule
from medAI.modeling import nn, torch
from medAI.modeling.prostnfound import ProstNFound
from medAI.modeling.setr import SETR
from medAI.modeling.registry import register_model, create_model


import numpy as np
import torch
import torch.nn as nn


import logging


class ProstNFoundModelInterface(nn.Module): 
    def forward(self, data: dict, include_postprocessed_heatmaps=False) -> dict:
        """Forward pass for ProstNFound-style models.

        Args:
            data: Dict containing at least the following keys:
                - 'bmode': B-mode ultrasound images, shape (B, C, H, W)
                - 'prostate_mask': Prostate masks, shape (B, 1, H, W)
                - 'needle_mask': Needle masks, shape (B, 1, H, W)
                - Optional additional keys for prompts
            include_postprocessed_heatmaps: If True, includes postprocessed heatmaps in the output dict

        Returns:
            data: Dict augmented with at least the following
                - 'cancer_logits': Cancer logits, shape (B, 1, H, W)
                - 'average_needle_heatmap_value': Average heatmap value within needle region, shape
                    (B,)
                - Optional: 
                    - 'image_level_classification_outputs': Outputs from image-level classification head
                    - 'cancer_probs': Postprocessed cancer probability heatmaps (if include_postprocessed_heatmaps is True)
        """
        raise NotImplementedError

    def get_params_groups(self):
        """Get parameter groups for optimizer setup.

        Returns:
            Tuple of three lists of parameters:
                - encoder_parameters: Parameters for the encoder (e.g., image encoder)
                - warmup_parameters: Parameters for warmup (e.g., classification heads)
                - cnn_parameters: Parameters for CNN components (if any)
        """
        from itertools import chain

        encoder_parameters = []
        warmup_parameters = self.parameters()
        cnn_parameters = []

        return encoder_parameters, warmup_parameters, cnn_parameters

    @property
    def device(self):
        return next(self.parameters()).device


class ProstNFoundWrapperForHeatmapModel(ProstNFoundModelInterface):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, data, include_postprocessed_heatmaps=False):
        bmode = data["bmode"].to(self.device)
        needle_mask = data["needle_mask"].to(self.device)
        prostate_mask = data["prostate_mask"].to(self.device)
        B = len(bmode)

        cancer_logits = self.model(bmode)
        data["cancer_logits"] = cancer_logits

        # compute predictions
        masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        mean_predictions_in_needle = []
        for j in range(B):
            mean_predictions_in_needle.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)
        data["average_needle_heatmap_value"] = mean_predictions_in_needle

        prostate_masks = prostate_mask > 0.5
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, prostate_masks)
        mean_predictions_in_prostate = []
        for j in range(B):
            mean_predictions_in_prostate.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)
        data["average_prostate_heatmap_value"] = mean_predictions_in_prostate

        if include_postprocessed_heatmaps:
            cancer_logits = data["cancer_logits"]
            heatmap = cancer_logits[0, 0].detach().sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample
            import cv2

            blurred = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.5)
            upsampled = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_LINEAR)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]

        return data

        return data


class ProstNFoundModelWrapper(ProstNFoundModelInterface):
    """Wraps a model to perform forward pass with ProstNFound style training

    Args:
        model: The model to wrap.
        mask_output_key: The key to use for the mask output (if the model outputs a dictionary of tensors)
    """

    def __init__(self, model: nn.Module, mask_output_key=None, model_runner=None):
        super().__init__()
        self.model = model
        self.mask_output_key = mask_output_key

        if isinstance(self.model, ProstNFound):
            logging.info(f"Model ProstNFound with prompts {self.model.prompts}")

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, include_postprocessed_heatmaps=False):
        # extracting relevant data from the batch
        bmode = data["bmode"].to(self.device)
        needle_mask = data["needle_mask"].to(self.device)
        prostate_mask = data["prostate_mask"].to(self.device)
        if "rf" in data:
            rf = data["rf"].to(self.device)
        else:
            rf = None

        B = len(bmode)

        # Wrapped forward pass
        if isinstance(self.model, ProstNFound):
            prompts = {}
            for prompt_name in self.model.prompts:
                prompts[prompt_name] = data[prompt_name].to(
                    device=self.device, dtype=bmode.dtype
                )
                if prompts[prompt_name].ndim == 1:
                    prompts[prompt_name] = prompts[prompt_name][:, None]

            outputs = self.model(
                bmode, rf, prostate_mask, needle_mask, output_mode="all", **prompts
            )
            cancer_logits = outputs["mask_logits"]
            image_level_classification_outputs = outputs["cls_outputs"]
            data["image_level_classification_outputs"] = (
                image_level_classification_outputs
            )
        else:
            model_outputs = self.model(bmode)
            if isinstance(model_outputs, dict):
                cancer_logits = model_outputs[self.mask_output_key]
            else:
                cancer_logits = self.model(bmode)

        cancer_logits = (
            cancer_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )
        data["cancer_logits"] = cancer_logits

        # compute predictions
        masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        mean_predictions_in_needle = []
        for j in range(B):
            mean_predictions_in_needle.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)
        data["average_needle_heatmap_value"] = mean_predictions_in_needle

        prostate_masks = prostate_mask > 0.5
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, prostate_masks)
        mean_predictions_in_prostate = []
        for j in range(B):
            mean_predictions_in_prostate.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)
        data["average_prostate_heatmap_value"] = mean_predictions_in_prostate

        if include_postprocessed_heatmaps:
            cancer_logits = data["cancer_logits"]
            heatmap = cancer_logits[0, 0].detach().sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample
            import cv2

            blurred = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.5)
            upsampled = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_LINEAR)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]

        return data

    def get_params_groups(self):
        if isinstance(self.model, SETR):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "head" in name:
                    warmup_parameters.append(param)
                else:
                    encoder_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif isinstance(self.model, ProstNFound):
            return self.model.get_params_groups()

        elif hasattr(self.model, "image_encoder"):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "image_encoder" in name:
                    encoder_parameters.append(param)
                else:
                    warmup_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif hasattr(self.model, "get_params_groups"):
            return self.model.get_params_groups()

        else:
            from itertools import chain

            encoder_parameters = []
            warmup_parameters = self.model.parameters()
            cnn_parameters = []

            return encoder_parameters, warmup_parameters, cnn_parameters


@register_model 
def prostnfound_wrapper_for_heatmap_model(*, base_model_cfg: dict):
    base_model = create_model(**base_model_cfg)
    return base_model