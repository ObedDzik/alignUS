import torch 
from torch import nn 


class Adapter(nn.Module):
    def __init__(self, feature_dim, adapter_dim, init_scale=1e-3):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(feature_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, feature_dim)
        self.act = nn.GELU()

        # initializations to make it close to identity function
        nn.init.uniform_(self.down_project.weight, -init_scale, init_scale)
        nn.init.uniform_(self.up_project.weight, -init_scale, init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return self.up_project(self.act(self.down_project(x))) + x


def freeze_non_adapter_layers(module: nn.Module, adapter_cls=Adapter): 
    for name, submodule in module.named_children(): 
        if not isinstance(submodule, adapter_cls): 
            for param in submodule.parameters(): 
                param.requires_grad = False 
        else: 
            freeze_non_adapter_layers(submodule, adapter_cls=adapter_cls)
