import logging
import os
from pathlib import Path
import torch

from medAI.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


def setup_and_load_checkpoint_dir(
    checkpoint_dir=None, dir=None, checkpoint_path="last.pth"
):
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / checkpoint_path
    else:
        assert dir is not None, "Either checkpoint_dir or dir must be provided"
        checkpoint_path = Path(dir) / "checkpoints" / checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint path: {checkpoint_path}")
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    else:
        logger.info(f"No checkpoint found at {checkpoint_path}")
        return None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


