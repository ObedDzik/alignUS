import torch


__all__ = ["MapLabelToSegmentationTarget"]


class MapLabelToSegmentationTarget:
    """
    Map labels to binary masks for weak segmentation tasks.
    """

    def __init__(
        self,
        reference_mask_key,
        label_key="label",
        output_key="weak_segmentation_target",
        ignore_index=-1,
    ):
        """
        Args:
            label_to_mask (dict): A dictionary mapping label values to binary mask values.
        """
        self.reference_mask_key = reference_mask_key
        self.label_key = label_key
        self.ignore_index = ignore_index
        self.output_key = output_key

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): A dictionary containing 'label' key with the label tensor.

        Returns:
            dict: The input dictionary with an added 'mask' key containing the binary mask tensor.
        """
        label = sample[self.label_key]
        ref_mask = sample[self.reference_mask_key]
        if ref_mask.ndim == 3:
            ref_mask = ref_mask[0]
        H, W = ref_mask.shape
        mask = torch.empty((H, W), dtype=torch.long)
        mask.fill_(self.ignore_index)
        mask[ref_mask.bool()] = label

        sample[self.output_key] = mask
        return sample
