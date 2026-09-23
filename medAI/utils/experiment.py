import argparse
from dataclasses import dataclass, asdict, is_dataclass
import datetime
import json
import sys
from typing import Literal
import os
from omegaconf import DictConfig, OmegaConf
from simple_parsing import subgroups
import torch.distributed
from wandb import Image
import torch
import logging
import wandb
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Literal
from submitit.helpers import DelayedSubmission
from submitit import SlurmExecutor
from filelock import FileLock


logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Experiment:
    """Handles the boiler plate of experiment setup."""

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str | None = 'slurm',
        log_mode: Literal["wandb", "textfile", "info"] = "wandb",
        distributed: bool = False,
        dist_url: str = "env://",
        conf=None,
        wandb_kwargs={},
        debug: bool | None = None,
        path_to_initial_checkpoint=None,
        resume=True,
    ):

        # setup output dir
        if checkpoint_dir == 'slurm': 
            if os.getenv("SLURM_JOB_ID") is not None: 
                checkpoint_dir = os.path.join("/checkpoint", os.environ["USER"], os.environ["SLURM_JOB_ID"])

        with FileLock(os.path.join(checkpoint_dir, 'lock')):
            if checkpoint_dir and os.path.exists(checkpoint_dir):
                logging.info(
                    f"Checkpoint directory exists: {checkpoint_dir}."
                )
                if resume: 
                    if 'output_dir.txt' in os.listdir(checkpoint_dir):
                        with open(os.path.join(checkpoint_dir, 'output_dir.txt')) as f: 
                            output_dir = f.read().strip()
                    logging.info(f"Using output dir {output_dir} found in checkpoint directory.")
                else:
                    logging.info(f"Resume is set to False. Ignoring output_dir.txt.")

        self.output_dir = output_dir
        print(f"Output dir: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.distributed = distributed
        self.dist_url = dist_url
        self.rank = None 
        self.world_size = None
        self._setup_distributed()

        # read and set debug mode
        if debug is None:
            self.debug = os.environ.get("DEBUG", False)
        else:
            self.debug = debug
        if self.debug:
            os.environ["WANDB_MODE"] = "disabled"
            log_mode = "info"

        # setup checkpoint dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if checkpoint_dir is not None:
            with FileLock(os.path.join(checkpoint_dir, 'lock')): # file lock to prevent multiple processes from trying to perform the same symlink
                if not os.path.exists(os.path.join(self.output_dir, "checkpoints")):
                    logging.info(
                        f"Symbolically linking checkpoint directory: {os.path.join(self.output_dir, 'checkpoints')} -> {checkpoint_dir}"
                    )
                    os.symlink(
                        checkpoint_dir,
                        os.path.join(self.output_dir, "checkpoints"),
                        target_is_directory=True,
                    )
        else:
            logging.info(f"Making checkpoint directory")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        with FileLock(os.path.join(self.checkpoint_dir, 'lock')):
            # convert config to object
            if isinstance(conf, argparse.Namespace):
                conf = OmegaConf.create(vars(conf))
            elif is_dataclass(conf):
                conf = OmegaConf.create(asdict(conf))
            elif isinstance(conf, DictConfig):
                ...
            else: 
                raise ValueError(f"Unknown type of conf: {type(conf)}")
            self.conf = conf
            OmegaConf.save(conf, os.path.join(self.output_dir, "run_config.yaml"))
            OmegaConf.save(conf, os.path.join(self.output_dir, "run_config_resolved.yaml"), resolve=True)
            with open(os.path.join(self.output_dir, "command.sh"), "w") as f:
                f.write(" ".join(sys.argv))
                f.flush()
            with open(os.path.join(self.checkpoint_dir, 'output_dir.txt'), 'w') as f: 
                f.write(self.output_dir)
                f.flush()

        self.log_mode = log_mode
        if self.rank == 0:
            # setup logging
            logging.basicConfig(
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(os.path.join(self.output_dir, "out.log")),
                ],
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            if self.log_mode == "wandb":
                logging.info(
                    f"Setting up wandb. Note that you can pass options to wandb through environment variables!"
                )
                self.wandb_run = wandb.init(
                    config=OmegaConf.to_object(conf), **wandb_kwargs
                )
                wandb.save(
                    os.path.join(self.output_dir, "run_config.yaml"),
                    base_path=self.output_dir,
                )
                wandb.save(
                    os.path.join(self.output_dir, "command.sh"),
                    base_path=self.output_dir,
                )
                wandb.run.log_code('.')
            else:
                self.wandb_run = None
        else:
            logging.basicConfig(
                handlers=[
                    logging.StreamHandler(sys.stdout),
                ],
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        if os.path.exists(os.path.join(self.checkpoint_dir, "last.pt")):
            logging.info(f"Loading last.pt from {self.checkpoint_dir}.")
            self.state = torch.load(
                os.path.join(self.checkpoint_dir, "last.pt"), map_location="cpu"
            )
            logging.info("\n".join([str(key) for key in self.state.keys()]))
        elif path_to_initial_checkpoint is not None:
            logging.info(
                f"Loading initial checkpoint from {path_to_initial_checkpoint}."
            )
            self.state = torch.load(path_to_initial_checkpoint, map_location="cpu")
        else:
            self.state = None

    def log_metrics(self, d: dict, step: bool = True):
        """Log a dict of metrics
        
        Args: 
            d: dictionary of metrics
            step: whether to log as a step or not (if is a step, logging to console/file is suppressed)
        """
        if self.rank != 0:
            return  # only log from rank 0
        if self.log_mode == "wandb":
            wandb.log(d)
        if not step:
            with open(os.path.join(self.output_dir, "metrics.txt"), "a") as f:
                f.write(str(d))
                f.flush()
                self.info(f"Metrics: {d}")

    def info(self, *args, **kwargs):
        """Alias of logging.info"""
        logging.info(*args, **kwargs)

    def save_checkpoint(self, state_dict, name: str = "last.pth"):
        if self.rank != 0:
            return  # only save from rank 0
        torch.save(state_dict, os.path.join(self.checkpoint_dir, name))

    def _setup_distributed(self): 
        # setup distributed
        if self.distributed:
            if os.environ.get("RANK") is None:
                if "SLURM_JOB_ID" in os.environ:
                    logging.info(
                        "Setting up distributed mode from SLURM environment variables."
                    )
                    os.environ["RANK"] = os.environ["SLURM_PROCID"]
                    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

            if os.environ.get("MASTER_ADDR") is None: 
                logging.info("Setting MASTER_PORT and MASTER_ADDR automatically.")
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "29500"

            rank = int(os.environ.get("RANK"))
            torch.cuda.set_device(rank)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

            torch.distributed.init_process_group(
                backend="nccl", init_method=self.dist_url
            )
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

            logging.info(f"Distributed mode active - {self.rank=} {self.world_size=}")
        else:
            logging.info("Distributed mode not active.")
            self.world_size = 1
            self.rank = 0


def format_dir_string(s: str) -> str:
    """Format the string to replace %d and %t with the current date and time"""
    new_output_dir = s.replace("%d", datetime.datetime.now().strftime("%Y-%m-%d"))
    new_output_dir = new_output_dir.replace(
        "%t", datetime.datetime.now().strftime("%H-%M-%S")
    )
    return new_output_dir


@dataclass
class Launcher:
    """Runs the function locally."""

    def __call__(self, func, *args, **kwargs):
        """Run the target function with the given arguments."""
        return func(*args, **kwargs)


class _Main:
    def __init__(self, run_dir, handle_preemption):
        self.run_dir = run_dir
        self.handle_preemption = handle_preemption
        self.argv = sys.argv

    def __call__(self, func, *args, **kwargs):
        sys.argv = (
            self.argv
        )  # this is a hack to make the wandb logger work and show original arguments

        # make the run directory, symbolically link the checkpoint path
        os.makedirs(self.run_dir, exist_ok=True)
        # with open(os.path.join(self.run_dir, "argv.txt"), "w") as f:
        #     f.write(" ".join(self.argv))

        checkpoint_dir = os.path.join(
            "/checkpoint", os.environ["USER"], os.environ["SLURM_JOB_ID"]
        )
        if not os.path.exists(os.path.join(self.run_dir, "checkpoints")):
            try:
                os.symlink(
                    checkpoint_dir,
                    os.path.join(self.run_dir, "checkpoints"),
                    target_is_directory=True,
                )
            except:
                pass

        # save the function in case we need to resubmit
        self.func = func
        self.args = args
        func(*args, **kwargs)

    def checkpoint(self, *args, **kwargs):
        if not self.handle_preemption:
            print(
                "Caught preemtion signal, and we are not set up to handle preemption. Exiting..."
            )
        return DelayedSubmission(
            _Main(self.run_dir, self.handle_preemption), self.func, *args, **kwargs
        )


@dataclass
class VectorSlurmLauncher(Launcher):
    """Submits the function to the vector cluster to run.

    Args:
        gpus: number of gpus to ask for - runs the job in parallel
            across this many processes.
        partition: Choose the desired GPU types
        qos: QOS to ask for - further left means less priority and smaller
            time limits but lets more jobs be submitted at once.
        mem_per_gpu: Memory (gigabytes) per gpu
        cpus_per_task: number of cpus per task
        output_dir: Output files and checkpoint directory will be created here
        handle_preemption: if set, the process will automaticall resumbit it self after preemtion
            OR timeout.
        num_tasks: number of tasks to spawn. If None, will be set to the number of GPUS
        setup: additional batch commands to run before the job starts
    """

    QOS_TIMES = {"normal": 16, "m": 12, "m2": 8, "m3": 4}

    gpus: int = 1
    partition: list[Literal["t4v2", "a40", "rtx6000"]] = field(
        default_factory=lambda: ["a40", "rtx6000", "t4v2"]
    )
    qos: Literal["normal", "m", "m2", "m3"] = "m2"
    mem_per_gpu: int = 16
    cpus_per_task: int = 4
    output_dir: str = ".submitit"
    handle_preemption: bool = False
    num_tasks: int | None = None
    setup: list[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        if self.num_tasks is None:
            self.num_tasks = self.gpus

    def __call__(self, func, *args, **kwargs):
        executor = SlurmExecutor(folder=self.output_dir, max_num_timeout=100)
        executor.update_parameters(
            gres=f"gpu:{self.gpus}",
            qos=self.qos,
            cpus_per_task=self.cpus_per_task,
            mem=f"{self.mem_per_gpu*self.gpus}G",
            partition=",".join(self.partition),
            signal_delay_s=60 * 4,
            stderr_to_stdout=False,
            setup=[
                "export TQDM_MININTERVAL=30",  # avoid too many updates to tqdm
                "export WANDB_RESUME=allow",
                "export WANDB_RUN_ID=$SLURM_JOB_ID",
            ]
            + self.setup,
            time=self.QOS_TIMES[self.qos] * 60,
            ntasks_per_node=self.gpus,
        )
        job = executor.submit(
            _Main(run_dir=self.output_dir, handle_preemption=self.handle_preemption),
            func,
            *args,
            **kwargs,
        )
        print(job.job_id)

    @staticmethod
    def add_args(parser): 
        parser.add_argument('--gpus', type=int, default=1, help='Number of gpus to use.')
        parser.add_argument('--partition', type=str, default="a40,rtx6000,t4v2", help='Partition to submit to.')
        parser.add_argument('--qos', type=str, default="m2", help='QOS to use.')
        parser.add_argument('--mem_per_gpu', type=int, default=16, help='Memory per gpu.')
        parser.add_argument('--cpus_per_task', type=int, default=4, help='CPUs per task.')

        # check if parser already has output_dir
        # if not, add it
        if not any(arg.dest == 'output_dir' for arg in parser._actions): 
            parser.add_argument('--output_dir', type=str, default=".submitit", help='Output directory.')
        parser.add_argument('--handle_preemption', action='store_true', help='Handle preemption.')
        parser.add_argument('--num_tasks', type=int, default=None, help='Number of tasks.')
        parser.add_argument('--setup', type=str, default=[], nargs='+', help='Setup commands.')

    @staticmethod
    def from_args(args): 
        return VectorSlurmLauncher(
            gpus=args.gpus,
            partition=args.partition.split(","),
            qos=args.qos,
            mem_per_gpu=args.mem_per_gpu,
            cpus_per_task=args.cpus_per_task,
            output_dir=args.output_dir,
            handle_preemption=args.handle_preemption,
            num_tasks=args.num_tasks,
            setup=args.setup
        )


LAUNCHERS = {"local": Launcher, "vc": VectorSlurmLauncher}
