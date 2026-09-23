"""
Miscellaneous models and utilities for fast prototyping.
"""


from medAI.modeling.registry import create_model, register_model
import torch 


@register_model
def medsam_hacked(bb="vit_small.medibot-v0", **kwargs): 
    model = create_model("medsam")
    new_image_encoder = create_model(bb, return_feature_map=True)
    embedding_dim = new_image_encoder.embed_dim

    class DummyImageEncoder(torch.nn.Module): 
        def __init__(self, image_encoder): 
            super().__init__()
            self.image_encoder = image_encoder
            self.neck = torch.nn.Conv2d(embedding_dim, 256, kernel_size=1)

        def forward(self, x): 
            x = self.image_encoder(x)
            x = self.neck(x)
            return x

    new_image_encoder = DummyImageEncoder(new_image_encoder)
    model.image_encoder = new_image_encoder
    return model