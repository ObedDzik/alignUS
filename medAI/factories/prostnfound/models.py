from warnings import warn

from omegaconf import OmegaConf
from medAI.modeling import create_model


def get_model(cfg, **override_kwargs):
    if isinstance(cfg['model'], str):
        warn("Using deprecated string model specification. Please use dict format like 'model: {{name: {cfg.model}}, **model_kw: ...}}'")
        model_kw = dict(
            name=cfg['model'],
            **cfg.get('model_kw', dict())
        )
        model_kw.update(override_kwargs)
        model = create_model(**model_kw)
        print("model_kw:", model_kw)
    else: 
        # default model implementation
        model_kw = cfg['model']
        model_kw.update(override_kwargs)
        model = create_model(
            **model_kw
        )
        print(f"Model cfg: {model_kw}")

    print(f"Model: {type(model)}")
    return model