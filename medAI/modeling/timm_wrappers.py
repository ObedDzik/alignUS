import timm
from medAI.modeling.registry import register_model


@register_model
def resnet18(*args, checkpoint=None, **kwargs): 
    return timm.create_model('resnet18', *args, **kwargs, checkpoint_path=checkpoint)


@register_model
def resnet50(**kwargs): 
    return timm.create_model('resnet50', **kwargs)