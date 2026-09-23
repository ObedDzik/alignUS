import os
import submitit
from medAI.utils.logging import get_default_log_dir
import argparse
from omegaconf import OmegaConf


class _Main:
    def __init__(self, cfg, task_fn): 
        self.cfg = cfg 
        self.task_fn = task_fn

    def checkpoint(self): 
        return submitit.helpers.DelayedSubmission(self)

    def __call__(self): 
        os.environ['WANDB_RUN_ID'] = os.environ["SLURM_JOB_ID"]
        os.environ['WANDB_RESUME'] = 'allow'

        return self.task_fn(self.cfg)


slurm_defaults = dict(
    nodes=1,
    slurm_job_name="tus-rec",
    cpus_per_task=10,
    tasks_per_node=1,
    slurm_gres="gpu:1",
    slurm_account="aip-medilab",
    timeout_min=4 * 60,
    mem_gb=128,
)


def launch_with_submitit(task_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=get_default_log_dir())
    parser.add_argument("--config", "-c", help="Path to yaml configuration file")
    parser.add_argument("--resume_from_wandb", metavar='PATH', help="Resume from Weights & Biases")
    parser.add_argument("--no-submitit", action='store_true', help="Run without submitit (locally)")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Overrides to config")

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if args.resume_from_wandb: 
        from wandb import Api
        api = Api() 
        run = api.run(args.resume_from_wandb)
        f = run.file('config_resolved.yaml')
        with f.download('/tmp', replace=True) as f:
            cfg = OmegaConf.load(f)

    else: 
        cfg = OmegaConf.create({"log_dir": args.log_dir})
        if args.config:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
        if args.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    if args.no_submitit:
        print("Running without submitit")
        return task_fn(cfg)

    slurm_cfg = cfg.get('slurm', {})
    slurm_cfg = OmegaConf.merge(OmegaConf.create(slurm_defaults), slurm_cfg)

    executor = submitit.AutoExecutor(folder=args.log_dir, max_num_timeout=25)
    executor.update_parameters(
        **slurm_cfg
    )
    # Pass your config path as argument
    job = executor.submit(_Main(cfg, task_fn))
    print(f"Submitted job: {job.job_id}")
    print(f"{job.paths.stdout} for stdout")


