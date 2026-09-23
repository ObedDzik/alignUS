from medAI.modeling.registry import register_model, create_model
import torch


@register_model 
def medibot_pretrained_model(*, model_cfg: dict, checkpoint_path=None, teacher_or_student='teacher', prefix='vit.'):
    model = create_model(**model_cfg)

    if checkpoint_path is None:
        return model

    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = state[teacher_or_student]
    state = {
        k.replace(prefix, ''): v for k, v in state.items() if k.startswith(prefix)
    }
    state = {
        k.replace("backbone.", ''): v for k, v in state.items() if k.startswith("backbone.")
    }

    print(model.load_state_dict(state, strict=False))
    print(f"Loaded {teacher_or_student} weights from {checkpoint_path}")
    return model
    