import torch 


def get_features_in_patch(image, patch_position_xyxy, feature_map):
    B, C, H, W = image.shape

    patch_position_mask = torch.zeros(B, 1, H, W, device=image.device)
    for i in range(B):
        xmin, ymin, xmax, ymax = patch_position_xyxy[i][
            0
        ]  # 0 because there is only one patch (for now)
        patch_position_mask[i, :, ymin:ymax, xmin:xmax] = 1

    npatches = feature_map.shape[2] * feature_map.shape[3]

    patch_position_mask = torch.nn.functional.interpolate(
        patch_position_mask,
        size=(int(npatches**0.5), int(npatches**0.5)),
        mode="nearest",
    )

    features_in_patch = (
        (feature_map * patch_position_mask).mean(dim=[2, 3])
    )

    return features_in_patch


class GetFeaturesInPatch(torch.nn.Module): 
    def forward(self, image, feature_map, patch_position_xyxy): 
        return get_features_in_patch(image, patch_position_xyxy, feature_map)