from dataclasses import dataclass
from medAI import registry
from medAI.factories.utils.build_dataloader import build_dataloader


@dataclass
class DataloaderConfig:
    transform: dict
    dataset: dict
    dataloader: dict


@registry.register("dataloader", "dataloader_dict_from_config")
def build_dataloaders_from_config(cfg: dict[str, DataloaderConfig]):

    # build dataloaders
    loaders = {}
    for loader_name, loader_cfg in cfg.items():
        transform = registry.build("transform", **loader_cfg.transform)
        dataset = registry.build("dataset", **loader_cfg.dataset, transform=transform)
        loader = build_dataloader(dataset, **loader_cfg.dataloader)
        loaders[loader_name] = loader

    return loaders