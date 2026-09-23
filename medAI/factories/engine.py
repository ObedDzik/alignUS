import logging
from omegaconf import OmegaConf
from medAI.engine.ssl_evaluation.kfold_nct_probing import KFoldNCTProbing
from medAI.engine.ssl_evaluation.linear_probing import (
    LinearProbing,
    FeatureMapLinearProbing,
)
from medAI import registry
from medAI.engine.ssl_evaluation.ssl_evaluation_ibot import (
    CIFAR10SSLEvaluator,
    DisabledSSLEvaluator,
    ProbingSSLEvaluatorForiBOT,
    SKLearnProbingSSLEvaluator,
    SSLEvaluatorNCT,
)


def build_kfold_nct_probing(**kwargs):
    return KFoldNCTProbing(**kwargs)


def build_linear_probing(**kwargs):
    return LinearProbing(**kwargs)


def build_patch_feature_linear_probing(**kwargs):
    return FeatureMapLinearProbing(**kwargs)


def build_probes_from_config(cfg, dataloaders_dict):
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, dict)

    probes = {}
    for name, kw in cfg.items():
        assert (
            "name" in kw
        ), f"Each probe config must have a 'name' field. Missing for {name}"

        train_loader_ref = kw.pop("train_loader_ref", None)
        val_loader_ref = kw.pop("val_loader_ref", None)

        if train_loader_ref is not None:
            kw["train_loader"] = dataloaders_dict[train_loader_ref]
        if val_loader_ref is not None:
            kw["val_loader"] = dataloaders_dict[val_loader_ref]
        if name == "kfold_nct_probing":
            output_adapter = lambda x: x["last_feature_map"]
            kw["output_adapter"] = output_adapter
        probe = registry.build("engine", **kw, device="cuda")
        probes[name] = probe

    return probes


def get_ibot_cnn_distillation_evaluator_v0(conf):
    if conf.evaluation.get("mode", "nct") == "nct":
        return SSLEvaluatorNCT(conf)
    elif conf.evaluation.get("mode", "nct") == "cifar10":
        return CIFAR10SSLEvaluator(conf)
    elif conf.evaluation.get("mode", "nct") == "sklearn":
        return SKLearnProbingSSLEvaluator(conf)
    elif conf.evaluation.get("mode", "nct") == "disabled":
        return DisabledSSLEvaluator(conf)
    else:
        raise ValueError(
            f"Invalid evaluation mode: {conf.evaluation.get('mode', 'nct')}"
        )


def get_ibot_cnn_distillation_evaluator_v1(conf):
    if "probes" in conf.evaluation and len(conf.evaluation.probes) > 0:
        # schema version 2
        logging.info(f"Building evaluator from config...")
        from medAI.factories.data.build_dataloaders_from_config import (
            build_dataloaders_from_config,
        )
        from medAI.factories.engine import build_probes_from_config

        eval_loaders = build_dataloaders_from_config(conf.evaluation.dataloaders)
        probes = build_probes_from_config(conf.evaluation.probes, eval_loaders)
        return ProbingSSLEvaluatorForiBOT(probes)
    else:
        # schema version 1
        return get_ibot_cnn_distillation_evaluator_v0(conf)