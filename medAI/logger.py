from pathlib import Path
from omegaconf import OmegaConf
from medAI.utils.distributed import is_main_process


def build_logger_from_cfg(cfg): 
    logger_cfg = cfg.get("logger", None)
    if logger_cfg is None: 
        return None 
    if not is_main_process():
        return None

    log_dir = Path(logger_cfg.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config-non-resolved.yaml")
    OmegaConf.save(cfg, log_dir / "config-resolved.yaml", resolve=True)

    if logger_cfg.logger_type == "wandb": 
        import wandb 
        wandb.init(
            project=logger_cfg.get("project", "medAI"),
            config=OmegaConf.to_object(cfg),
            **logger_cfg.wandb_init_kwargs,
        )
        wandb.save(f"{log_dir}/*.yaml", base_path=log_dir)
        # wandb.save(str(log_dir / "config-resolved.yaml"), base_path=log_dir)
        # wandb.save(str(log_dir / "config-non-resolved.yaml"), base_path=log_dir)
        return wandb.log
        