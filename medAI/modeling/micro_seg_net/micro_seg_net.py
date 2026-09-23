import argparse
from torch import nn 
from huggingface_hub import PyTorchModelHubMixin
from .MicroSegNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from .MicroSegNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os
from medAI.modeling.registry import register_model
import torch 


_MODEL_DIR = '/h/pwilson/projects/medAI/model_checkpoints'


class MicroSegNet(nn.Module, PyTorchModelHubMixin): 
    def __init__(self): 
        super().__init__()
        args = argparse.Namespace()
        args.img_size = 224
        args.vit_patches_size = 16 

        conf = CONFIGS_ViT_seg['R50-ViT-B_16']
        conf.n_classes = 1
        conf.n_skip = 3 
        conf.patches.size = (16, 16)
        conf.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

        self.net = ViT_seg(conf, img_size=args.img_size, num_classes=conf.n_classes)

    def forward(self, x): 
        return self.net(x)[0]


@register_model
def micro_seg_net(): 

    args = argparse.Namespace()
    args.img_size = 224
    args.vit_patches_size = 16 

    conf = CONFIGS_ViT_seg['R50-ViT-B_16']
    conf.n_classes = 1
    conf.n_skip = 3 
    conf.patches.size = (16, 16)
    conf.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    model = ViT_seg(conf, img_size=args.img_size, num_classes=conf.n_classes)

    from huggingface_hub import hf_hub_download
    checkpoint = hf_hub_download(
        'pfrwilson/micro_seg_net', "MicroSegNet.pth",
    )
    if os.path.exists(checkpoint): 
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    else: 
        print(f"Checkpoint {checkpoint} not found. Using uninitialized model.")

    old_forward = model.forward
    def forward(*args, **kwargs): 
        return old_forward(*args, **kwargs)[0]
    model.forward = forward


    return model