import logging
import math
from pathlib import Path
import sys
from medAI.engine.ssl_evaluation.kfold_nct_probing import KFoldNCTProbing
from medAI.factories.utils.build_dataloader import build_dataloader
from medAI.utils import dino_utils as utils
import torch
from medAI.optimizer import build_optimizer_v0
from medAI.modeling.ibot.wrappers import MultiCropWrapper
from medAI.modeling.vision_transformer import VisionTransformer
from hydra.utils import instantiate
import medAI
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from medAI.modeling.ibot.head import DINOHead
from medAI.losses.dino_loss import DINOLoss
from medAI.utils.distributed import init_distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from medAI.logger import build_logger_from_cfg
from torch.optim import AdamW
from medAI.engine.ssl_evaluation.linear_probing import FeatureMapLinearProbing
from medAI.engine.ssl_evaluation.linear_probing import LinearProbing
from medAI.utils.checkpoint import setup_and_load_checkpoint_dir
from medAI.utils.reproducibility import set_global_seed
from medAI import registry


def main(cfg):

    set_global_seed(cfg.trainer.get("seed", 42))

    # setup logging
    logging.basicConfig(level=logging.INFO)

    init_distributed()

    dir = Path(cfg.trainer.dir)
    state = setup_and_load_checkpoint_dir(
        dir=cfg.trainer.dir, checkpoint_path="last.pth"
    )
    logger = build_logger_from_cfg(cfg)

    # build dataloaders
    loaders = {}
    for loader_name, loader_cfg in cfg.data.items():
        transform = registry.build("transform", **loader_cfg.transform)
        dataset = registry.build("dataset", **loader_cfg.dataset, transform=transform)
        loader = build_dataloader(dataset, **loader_cfg.dataloader)
        loaders[loader_name] = loader

    # build wrapped models
    student_backbone = registry.build("model", **cfg.backbone).cuda()
    student = MultiCropWrapper(
        student_backbone,
        DINOHead(in_dim=student_backbone.embed_dim, out_dim=cfg.loss.out_dim),
        backbone_output_adapter=lambda x: x["pre_logits_clstoken"],
    ).cuda()
    teacher_backbone = registry.build("model", **cfg.backbone).cuda()
    teacher = MultiCropWrapper(
        teacher_backbone,
        DINOHead(in_dim=teacher_backbone.embed_dim, out_dim=cfg.loss.out_dim),
        backbone_output_adapter=lambda x: x["pre_logits_clstoken"],
    ).cuda()
    teacher.load_state_dict(student.state_dict())
    # after building student/teacher, before DDP
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_without_ddp = teacher
    # teacher = DDP(teacher, device_ids=[torch.cuda.current_device()])
    student = DDP(student, device_ids=[torch.cuda.current_device()])
    dino_loss = DINOLoss(
        **cfg.loss,
    ).cuda()

    if state is not None:
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        teacher_without_ddp.load_state_dict(state["teacher_without_ddp"])
        dino_loss.load_state_dict(state["dino_loss"])

    # setup optimizer
    scaled_lr = 0.0005 * cfg.trainer.batch_size * utils.get_world_size() / 256

    pg1, pg2 = get_regularized_vs_not_regularized_params(student)
    param_groups = [
        {"params": pg1, "weight_decay": cfg.trainer.weight_decay},
        {"params": pg2, "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        param_groups,
        lr=scaled_lr,
    )
    wd_scheduler = utils.cosine_scheduler(
        base_value=cfg.trainer.weight_decay,
        final_value=cfg.trainer.weight_decay_end,
        epochs=cfg.trainer.epochs,
        niter_per_ep=len(loaders["ssl"]),
    )
    lr_scheduler = utils.cosine_scheduler(
        base_value=scaled_lr,
        final_value=cfg.trainer.lr_end,
        epochs=cfg.trainer.epochs,
        niter_per_ep=len(loaders["ssl"]),
        warmup_epochs=cfg.trainer.warmup_epochs,
    )
    momentum_schedule = utils.cosine_scheduler(
        base_value=cfg.trainer.momentum_teacher,
        final_value=cfg.trainer.momentum_teacher_end,
        epochs=cfg.trainer.epochs,
        niter_per_ep=len(loaders["ssl"]),
    )

    fp16_scaler = None
    if cfg.trainer.use_fp16:
        fp16_scaler = torch.GradScaler()

    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        if fp16_scaler is not None and "fp16_scaler" in state:
            fp16_scaler.load_state_dict(state["fp16_scaler"])

    probes = {}
    for name, probe_cfg in cfg.probes.items():
        kw = OmegaConf.to_object(probe_cfg)
        assert isinstance(kw, dict)
        if kw.pop("train_loader_ref", None) is not None:
            kw["train_loader"] = loaders[kw["train_loader_ref"]]
        if kw.pop("val_loader_ref", None) is not None:
            kw["val_loader"] = loaders[kw["val_loader_ref"]]
        if kw["name"] == "kfold_nct_probing":
            output_adapter = lambda x: x["last_feature_map"]
            kw["output_adapter"] = output_adapter
        probe = registry.build("engine", **kw, model=teacher_backbone, device="cuda")
        probes[name] = probe

    start_epoch = 0 if state is None else state["epoch"]

    
class DINOEngine:
    def __init__(
        self,
        student,
        teacher,
        teacher_without_ddp,
        dino_loss,
        dataloader,
        optimizer,
        lr_schedule, 
        wd_schedule,
        momentum_schedule,
        logger=None,
        use_fp16=False,
        clip_grad=0,
        freeze_last_layer=1,
        epochs=100,
        start_epoch=0,
        probing_interval=10,
        probes={},
        save_checkpoint_every=None,
        dir=None
    ):
        self.student = student
        self.teacher = teacher
        self.teacher_without_ddp = teacher_without_ddp
        self.dino_loss = dino_loss
        self.optimizer = optimizer
        self.data_loader = dataloader
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.momentum_schedule = momentum_schedule
        self.logger = logger
        self.use_fp16 = use_fp16
        self.clip_grad = clip_grad
        self.freeze_last_layer = freeze_last_layer
        self.epochs = epochs
        self.fp16_scaler = torch.GradScaler(enabled=use_fp16)
        self.start_epoch = start_epoch
        self.probing_interval = probing_interval
        self.probes = probes

    def run(self): 
        for epoch in range(self.start_epoch, self.epochs):

            if epoch % self.probing_interval == 0:
                # probing
                metrics = {}
                for name, probe in self.probes.items():
                    results = probe.run()
                    metrics.update({f"{name}/{k}": v for k, v in results.items()})

                print(f"Epoch {epoch - 1} probing results: {metrics}")
                metrics["epoch"] = epoch - 1
                if self.logger is not None:
                    self.logger(metrics)

            torch.save(
                {
                    "student": self.student.state_dict(),
                    "teacher": self.teacher.state_dict(),
                    "teacher_without_ddp": self.teacher_without_ddp.state_dict(),
                    "dino_loss": self.dino_loss.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "fp16_scaler": (
                        self.fp16_scaler.state_dict() if self.fp16_scaler is not None else None
                    ),
                },
                dir / "checkpoints" / "last.pth",
            )
            if (
                self.cfg.trainer.save_checkpoint_every is not None
                and (epoch + 1) % self.cfg.trainer.save_checkpoint_every == 0
            ):
                torch.save(
                    {
                        "student": student.state_dict(),
                        "teacher": teacher.state_dict(),
                        "teacher_without_ddp": teacher_without_ddp.state_dict(),
                        "dino_loss": dino_loss.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "fp16_scaler": (
                            fp16_scaler.state_dict() if fp16_scaler is not None else None
                        ),
                    },
                    dir / "checkpoints" / f"checkpoint-{epoch:04}.pth",
                )

    def train_one_epoch(self, epoch):
        # in the epoch loop
        if hasattr(self.data_loader, "sampler") and hasattr(self.data_loader.sampler, "set_epoch"):
            self.data_loader.sampler.set_epoch(epoch)

        metric_logger = utils.MetricLogger(delimiter="  ")

        header = "Epoch: [{}/{}]".format(epoch, self.epochs)
        for it, data in enumerate(metric_logger.log_every(self.data_loader, 10, header)):
            # update weight decay and learning rate according to their schedule
            it = len(self.data_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[it]

            # move images to gpu
            images = data["image"]
            images = [im.cuda(non_blocking=True) for im in images]
            # teacher and student forward passes + compute dino loss
            with torch.autocast('cuda', enabled=self.use_fp16):
                with torch.no_grad():
                    teacher_output = self.teacher(
                        images[:2]
                    )  # only the 2 global views pass through the teacher
                student_output = self.student(images)
                loss = self.dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            # student update
            self.optimizer.zero_grad()
            param_norms = None
            if self.fp16_scaler is None:
                loss.backward()
                if self.clip_grad:
                    param_norms = utils.clip_gradients(self.student, self.clip_grad)
                utils.cancel_gradients_last_layer(epoch, self.student, self.freeze_last_layer)
                self.optimizer.step()
            else:
                self.fp16_scaler.scale(loss).backward()
                if self.clip_grad:
                    self.fp16_scaler.unscale_(
                        self.optimizer
                    )  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(self.student, self.clip_grad)
                utils.cancel_gradients_last_layer(epoch, self.student, self.freeze_last_layer)
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = self.momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(
                    self.student.module.parameters(), self.teacher_without_ddp.parameters()
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=self.optimizer.param_groups[0]["weight_decay"])

            if self.logger is not None:
                self.logger(
                    {
                        "epoch": epoch,
                        "loss": loss.item(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "wd": self.optimizer.param_groups[0]["weight_decay"],
                    }
                )

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_regularized_vs_not_regularized_params(model):
    named_parameters = list(model.named_parameters())
    regularized = []
    not_regularized = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return regularized, not_regularized
