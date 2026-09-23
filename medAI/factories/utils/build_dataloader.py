from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import logging 


logger = logging.getLogger(__name__)


__all__ = ['build_dataloader']


def build_dataloader(dataset, distributed=None, shuffle=False, **kwargs):
    logger.info(f"Building dataloader from dataset of size {len(dataset)}, type {type(dataset)}")

    if distributed is None:
        distributed = dist.is_initialized()

    if distributed: 
        logger.info("Using distributed sampler for dataloader.")
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = None 
    else: 
        sampler = None

    return DataLoader(dataset, sampler=sampler, shuffle=shuffle, **kwargs)
