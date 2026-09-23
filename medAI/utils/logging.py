import os
from torch import distributed as dist 
import logging 
import sys 


def setup_logging(dir):
    os.makedirs(dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    file_handler = logging.FileHandler(
        os.path.join(dir, f"experiment_rank-{rank}.log"),
    )
    file_handler.setLevel(logging.INFO)

    for handler in logging.getLogger().handlers:
        logging.getLogger().removeHandler(handler)
    
    logging.basicConfig(handlers=[stream_handler, file_handler], level=logging.INFO)

