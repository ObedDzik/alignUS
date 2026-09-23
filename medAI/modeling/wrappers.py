import torch 


class TimmCNNWrapperForFeatures(torch.nn.Module):
    def __init__(self, cnn_model, pool=True):
        super().__init__()
        self.cnn_model = cnn_model
        self.pool = pool

    def forward(self, image):
        feature_map = self.cnn_model.forward_features(image)
        if self.pool:
            feature_map = torch.nn.functional.adaptive_avg_pool2d(feature_map, 1)[..., 0, 0]
        return feature_map


class VitWrapperForFeatureMaps(torch.nn.Module): 
    def __init__(self, vit): 
        super().__init__()
        self.vit = vit

    def forward(self, image): 
        B, C, H, W = image.shape
        vit_feature_map = self.vit(image, return_all_tokens=True)[:, 1:, :]
        B, npatches, C = vit_feature_map.shape
        vit_feature_map = vit_feature_map.reshape(
            B, int(npatches**0.5), int(npatches**0.5), C
        )
        vit_feature_map = vit_feature_map.permute(0, 3, 1, 2)  # B, C, H, W

        return vit_feature_map