from argparse import ArgumentParser


def get_submitit_parser(): 
    parser = ArgumentParser()
    parser.add_argument("--output_dir", '-d')
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--qos", default='m3')
    parser.add_argument("--timeout", default=4, type=int, help="Duration of the job in hours")
    parser.add_argument("--mem_gb", default=64, type=int, help="Memory in GB")
    
    return parser 


class Main: 
    def __init__(self, fn, args): 
        self.fn = fn
        self.args = args
    
    def __call__(self): 
        return self.fn(self.args)

    def checkpoint(self): 
        import submitit
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.fn, self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)


def get_executor(args): 
    import submitit
    executor = submitit.AutoExecutor(folder=args.output_dir, max_num_timeout=10)
    executor.update_parameters(
        mem_gb=64,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,
        cpus_per_task=8,
        timeout_min=2800,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
    )
    return executor


