from typing import Iterable, Mapping
import torch 


def convert_batch(batch, device): 
    primitives = (bool, str, int, float, type(None))
    if type(batch) in primitives:
        return batch
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, Mapping): 
        return {k: convert_batch(v, device) for k, v in batch.items()}
    elif isinstance(batch, Iterable):
        return [convert_batch(v, device) for v in batch]
    else:
        return batch


def add_prefix(d, prefix, sep="/"): 
    return {prefix + sep + k: v for k, v in d.items()}