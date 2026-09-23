import argparse
from itertools import chain
import os
import sys
import math

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from medAI.modeling.transformer import TransformerEncoder
import medAI.utils.ibot_dino_utils as utils
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import rich
from omegaconf import DictConfig, OmegaConf
import logging
from torch import distributed as dist
import dotenv
from dataclasses import dataclass, field

from medAI.losses.dino_loss import DINOLoss
from medAI.modeling.ibot.head import DINOHead
from medAI.losses.ibot_loss import iBOTLoss
from medAI.factories.engine import get_ibot_cnn_distillation_evaluator_v1 as get_ibot_cnn_distillation_evaluator
from medAI.modeling.ibot.wrappers import MultiCropWrapper
from medAI.factories.data.ibot_dataloaders import get_multicrop_ssl_dataloader_from_config
from medAI.utils.distributed import init_distributed, get_rank, get_world_size
from medAI.factories.optimizer import build_optimizer_v1, OptimizerCfg


dotenv.load_dotenv()


def train_ibot(conf):

    # ========= setup =======================
    os.makedirs(conf.output_dir, exist_ok=True)
    os.makedirs(conf.checkpoint_dir, exist_ok=True)
    OmegaConf.save(
        conf, os.path.join(conf.output_dir, "config-non-resolved.yaml"), resolve=False
    )
    OmegaConf.save(conf, os.path.join(conf.output_dir, "config-resolved.yaml"), resolve=True)

    init_distributed()
    print(f"{get_rank()=}, {get_world_size()=}")
    utils.fix_random_seeds(conf.seed)

    if utils.is_main_process():
        formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] %(name)s: %(message)s')
        logging.getLogger().setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler2 = logging.FileHandler(os.path.join(conf.output_dir, "experiment.log"))
        handler.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().addHandler(handler2)
        logging.getLogger().setLevel(logging.INFO)
        conf.wandb.id = str(conf.wandb.id)
        wandb.init(
            project="iBOT",
            config=OmegaConf.to_container(conf, resolve=True),  # type:ignore
            resume="allow",
            job_type="ssl_pretraining",
            **conf.wandb,
        )
        wandb.save(
            os.path.join(conf.output_dir, "*.yaml"), base_path=conf.output_dir
        )
    else: 
        logging.getLogger().setLevel(logging.CRITICAL)
    torch.backends.cudnn.benchmark = True

    # =========== Build the experiment

    DISABLE_IBOT_LOSS = conf.get('disable_ibot_loss', False)
    DISABLE_CNN_DISTILLATION = conf.get('disable_cnn_distillation', False)

    if conf.get('schema_version', 1.0) == 1.0: 
        data_loader = get_multicrop_ssl_dataloader_from_config(conf)
        prepare_batch_fn = None 
    else: 
        from hydra.utils import instantiate
        data_loader = instantiate(conf.ssl_loader)
        def _prepare_batch_fn(batch):
            images = batch['image']
            labels = torch.tensor([0] * len(images[0])).long()
            masks = batch['masks']
            return images, labels, masks
        prepare_batch_fn = _prepare_batch_fn

    vit_student, vit_teacher, cnn_student, cnn_teacher, predictor = get_models(conf)
    vit_student, vit_teacher, vit_teacher_without_ddp = setup_for_ddp(
        vit_student, vit_teacher
    )
    cnn_student, cnn_teacher, cnn_teacher_without_ddp = setup_for_ddp(
        cnn_student, cnn_teacher
    )
    predictor = predictor.cuda() if predictor is not None else None
    predictor = nn.parallel.DistributedDataParallel(
        predictor, device_ids=[dist.get_rank()], broadcast_buffers=False
    ) if predictor is not None else None

    teacher_without_ddp = nn.ModuleDict(
        {
            "vit": vit_teacher_without_ddp,
            "cnn": cnn_teacher_without_ddp,
        }
    )
    student = nn.ModuleDict(
        {
            "vit": vit_student,
            "cnn": cnn_student,
        }
    )
    teacher = nn.ModuleDict(
        {
            "vit": vit_teacher,
            "cnn": cnn_teacher,
        }
    )
    logging.info(
        f"Student and Teacher are built: {vit_student.__class__}, {cnn_student.__class__}"
    )

    # ============ preparing loss ... ============
    _build_dino_loss = lambda: DINOLoss(
        conf.model.n_prototypes,
        n_global_crops=conf.transform.global_crops_number + conf.transform.local_crops_number,
        warmup_teacher_temp=conf.loss.warmup_teacher_temp,
        teacher_temp=conf.loss.teacher_temp,
        warmup_teacher_temp_epochs=conf.loss.warmup_teacher_temp_epochs,
        nepochs=conf.epochs,
    )
    cnn_dino_loss = _build_dino_loss()

    if conf.loss.get('cross_loss_type', 'dino') == 'dino':
        cross_dino_loss = _build_dino_loss()
    elif conf.loss.get('cross_loss_type', 'dino') == 'mse':
        class MSELossWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse_loss = nn.MSELoss()
            def forward(self, x, y, epoch):
                return self.mse_loss(x, y)
        cross_dino_loss = MSELossWrapper()
    else: 
        raise ValueError(f"Unknown cross distillation loss type {conf.loss.cross_loss_type}")

    if DISABLE_IBOT_LOSS:
        vit_loss = _build_dino_loss()
    else:
        vit_loss = iBOTLoss(
            conf.model.n_prototypes,
            conf.model.n_prototypes,
            conf.transform.global_crops_number,
            conf.transform.local_crops_number,
            conf.loss.warmup_teacher_temp,
            conf.loss.teacher_temp,
            conf.loss.warmup_teacher_patch_temp,
            conf.loss.teacher_patch_temp,
            conf.loss.warmup_teacher_temp_epochs,
            conf.epochs,
            lambda1=conf.loss.lambda1,
            lambda2=conf.loss.lambda2,
            mim_start_epoch=conf.loss.pred_start_epoch,
        ).cuda()

    _ = vit_loss.cuda(), cnn_dino_loss.cuda(), cross_dino_loss.cuda()
    logging.info(f"Setup losses.")

    # ============ preparing optimizer ... ============
    models_for_optimization = nn.ModuleDict(
        {
            "vit": vit_student,
            "cnn": cnn_student,
        }
    )
    if predictor is not None:
        models_for_optimization["predictor"] = predictor
    optimizer, lr_schedulers, wd_schedulers, momentum_schedule = build_optimizer_v1(
        models_for_optimization,
        len(data_loader),
        OptimizerCfg(**conf.optimizer),
        conf.batch_size_per_gpu,
        conf.epochs,
    )
    fp16_scaler = torch.GradScaler('cuda') if conf.use_fp16 else None 
    
    logging.info(f"Loss, optimizer and schedulers ready.")

    # ============ setup additional evaluation ===============
    ssl_evaluator = get_ibot_cnn_distillation_evaluator(conf)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "best_score": 0.0}
    restore_variables = {}
    restore_variables["student"] = student
    restore_variables["teacher"] = teacher

    # prefer to resume from latest checkpoint in output directory
    if not conf.get('no_resume') and os.path.exists(p := os.path.join(conf.checkpoint_dir, "checkpoint.pth")):
        logging.info(f"Loading state from {p}")
        utils.restart_from_checkpoint(
            p,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=vit_loss,
        )
    elif conf.load_from is not None:
        load_path = os.path.join(conf.load_from, "checkpoint.pth")
        assert os.path.exists(load_path), f"Load path {load_path} does not exist."
        logging.info(f"Loading state from {load_path}")
        utils.restart_from_checkpoint(
            load_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=vit_loss,
        )
    elif (p := conf.load_model_from) is not None:
        assert os.path.exists(p), f"Load path {p} does not exist."
        logging.info(f"Loading models from {p}")
        utils.restart_from_checkpoint(
            p, run_variables=None, student=student, teacher=teacher, ibot_loss=vit_loss
        )
    start_epoch = to_restore["epoch"]
    best_score = to_restore["best_score"]

    logging.info("Starting iBOT training!")
    for epoch in range(start_epoch, conf.epochs):

        if hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)
        if hasattr(data_loader.dataset, "set_epoch"):
            data_loader.dataset.set_epoch(epoch)

        # ======== run and log probing results ========
        if not conf.get('skip_probing', False) and epoch % conf.evaluation.probing_freq == 0:
            probing_results = ssl_evaluator(
                teacher_without_ddp,
                epoch - 1,
            )

            if dist.get_rank() == 0:
                logging.info(f"Probing results: {probing_results}")
                wandb.log({"epoch": epoch - 1, **probing_results})
                if conf.get("monitored_metric") is not None:
                    if probing_results[conf.monitored_metric] > best_score:
                        logging.info(
                            f"New best score: {probing_results[conf.monitored_metric]}"
                        )
                        best_score = probing_results[conf.monitored_metric]
                        save_dict = {
                            "student": student.state_dict(),
                            "teacher": teacher.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch + 1,
                            "best_score": best_score,
                            "args": conf,
                            "ibot_loss": vit_loss.state_dict(),
                        }
                        if fp16_scaler is not None:
                            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
                        utils.save_on_master(
                            save_dict, os.path.join(conf.checkpoint_dir, "best.pth")
                        )

        # ============ training one epoch of iBOT ... ============
        logging.info(f"EPOCH {epoch}")
        train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            vit_loss,
            cnn_dino_loss,
            cross_dino_loss,
            data_loader,
            optimizer,
            lr_schedulers,
            wd_schedulers,
            momentum_schedule,
            epoch,
            fp16_scaler,
            conf,
            prepare_batch_fn=prepare_batch_fn,
            predictor=predictor,
            use_masks=not DISABLE_IBOT_LOSS,
            use_cnn_distillation=not DISABLE_CNN_DISTILLATION,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_score": best_score,
            "args": conf,
            "ibot_loss": vit_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(
            save_dict, os.path.join(conf.checkpoint_dir, "checkpoint.pth")
        )
        if conf.saveckp_freq and ((epoch + 1) % conf.saveckp_freq == 0) and epoch:
            utils.save_on_master(
                save_dict,
                os.path.join(conf.checkpoint_dir, f"checkpoint{epoch:04}.pth"),
            )


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    vit_distillation_loss: iBOTLoss | DINOLoss,
    cnn_distillation_loss: DINOLoss,
    cross_distillation_loss: nn.Module,
    data_loader,
    optimizer,
    lr_schedulers: list[list[float]],
    wd_schedulers: list[list[float]],
    momentum_schedule,
    epoch, 
    fp16_scaler,
    conf,
    predictor=None, 
    prepare_batch_fn=None,
    use_masks=True,
    use_cnn_distillation=True,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, conf.epochs)

    teacher.train()
    student.train()

    for it, data in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        images, labels, masks = prepare_batch_fn(data) if prepare_batch_fn is not None else data
        if conf.get("debug") and (it > 10):
            break

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        lrs_and_wds_for_logging = update_optimizer(
            it, optimizer, lr_schedulers, wd_schedulers
        )

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]
        batch_size = len(images[0])
        num_views = len(images)

        with torch.cuda.amp.autocast(fp16_scaler is not None):  # type:ignore

            # get global views
            losses_dict_for_logging = {}

            with torch.no_grad():
                vit_teacher_output = teacher["vit"](
                    images[: conf.transform.global_crops_number], return_all_tokens=True
                )

            vit_teacher_output_cls = vit_teacher_output[:, 0, :]
            vit_teacher_output_patch = vit_teacher_output[:, 2:, :]

            vit_student_output_dict = student["vit"](
                images[: conf.transform.global_crops_number],
                mask=masks[: conf.transform.global_crops_number] if (use_masks and epoch >= conf.loss.pred_start_epoch) else None,
                return_all_tokens=True,
                return_backbone_feat=True
            )
            vit_student_output_backbone = vit_student_output_dict['backbone_output']
            vit_student_output = vit_student_output_dict['head_output']
            vit_student_output_cls = vit_student_output[:, 0, :]
            vit_student_output_register = vit_student_output[:, 1, :]
            vit_student_output_patch = vit_student_output[:, 2:, :]

            # get local views
            if len(images) > conf.transform.global_crops_number:
                vit_student_local_view_output = student["vit"](
                    images[conf.transform.global_crops_number :], return_all_tokens=True
                )
                vit_student_local_cls = vit_student_local_view_output[:, 0, :]
                vit_student_local_register = vit_student_local_view_output[:, 1, :]
            else:
                vit_student_local_cls = None
                vit_student_local_register = None

            if isinstance(vit_distillation_loss, DINOLoss):
                loss_vit = vit_distillation_loss.forward(
                    vit_student_output_cls,
                    vit_teacher_output_cls,
                    epoch,
                )
            else: 
                loss_vit = vit_distillation_loss(
                    (vit_student_output_cls, vit_student_output_patch),
                    (vit_teacher_output_cls, vit_teacher_output_patch),
                    vit_student_local_cls,
                    masks,
                    epoch,
                )["loss"]
            
            losses_dict_for_logging["vit_loss"] = loss_vit.item()

            # get CNN views
            if use_cnn_distillation: 
                with torch.no_grad():
                    cnn_teacher_output_dict = teacher["cnn"](
                        images[: conf.transform.global_crops_number],
                        return_all_tokens=False,
                        return_backbone_feat=True, 
                    )
                    cnn_teacher_output = cnn_teacher_output_dict['head_output']
                    cnn_teacher_backbone_output = cnn_teacher_output_dict['backbone_output']

                cnn_student_output = student["cnn"](images, return_all_tokens=False)

                loss_cnn = cnn_distillation_loss(cnn_student_output, cnn_teacher_output, epoch)

                CROSS_DISTILLATION_MODE = conf.get('cross_distillation_mode', 'register')

                if CROSS_DISTILLATION_MODE == 'register':    
                    if vit_student_local_register is not None:
                        vit_student_output_register = torch.cat(
                            (vit_student_output_register, vit_student_local_register), dim=0
                        )
                    assert len(vit_student_output_register) == batch_size * num_views
                    
                    loss_cross = cross_distillation_loss(
                        vit_student_output_register, cnn_teacher_output, epoch
                    )
                
                elif CROSS_DISTILLATION_MODE == 'predictor': 
                    assert predictor is not None, "Predictor must be defined for predictor-based cross distillation."
                    
                    if conf.cross_distillation_target == "cnn_backbone": 
                        cnn_target = cnn_teacher_backbone_output
                    elif conf.cross_distillation_target == "cnn_head":
                        cnn_target = cnn_teacher_output
                    else: 
                        raise ValueError(f"Unknown cross distillation target {conf.cross_distillation_target}")

                    vit_to_cnn_pred = predictor(vit_student_output_backbone) # N x D_cnn
                    loss_cross = cross_distillation_loss(
                        vit_to_cnn_pred, cnn_target, epoch
                    )

                else: 
                    raise ValueError(f"Unknown cross distillation mode {CROSS_DISTILLATION_MODE}")
                    
                a1 = conf.loss.vit_loss_weight
                a2 = conf.loss.cnn_loss_weight
                a3 = conf.loss.cross_loss_weight
                loss = (
                    (loss_vit * a1 + loss_cnn * a2 + loss_cross * a3) * 3
                )
                losses_dict_for_logging["cnn_loss"] = loss_cnn.item()
                losses_dict_for_logging["cross_loss"] = loss_cross.item()
            
            else: 
                loss = loss_vit

            losses_dict_for_logging["total_loss"] = loss.item()

            if wandb.run is not None:
                wandb.log(
                    {
                        **losses_dict_for_logging,
                        **lrs_and_wds_for_logging,
                    }
                )

        losses_across_batch = utils.concat_all_gather(loss.view(1))
        if any([not math.isfinite(loss_.item()) for loss_ in losses_across_batch]):
            logging.info("Warning: NaN value encountered in loss")
            continue

        # student update
        optimizer.zero_grad()
        param_norms = None

        model_for_optim = nn.ModuleDict(
            dict(vit=student["vit"], cnn=student["cnn"])
        )
        if predictor is not None:
            model_for_optim["predictor"] = predictor

        if fp16_scaler is None:
            loss.backward()
            if conf.optimizer.clip_grad:
                param_norms = utils.clip_gradients(model_for_optim, conf.optimizer.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model_for_optim, conf.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if conf.optimizer.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model_for_optim, conf.optimizer.clip_grad)
            # utils.cancel_gradients_last_layer(epoch, model_for_optim, conf.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        # common params
        params_student, params_teacher = get_common_parameters(
            nn.ModuleDict({"vit": student["vit"].module, "cnn": student["cnn"].module}),
            teacher_without_ddp,
        )
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_student, params_teacher):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        # if param_norms is not None:
        #     metric_logger.update(param_norms=param_norms)

    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return return_dict


def get_models(conf: DictConfig):
    from medAI.factories.ibot.models import get_model as get_backbone

    student_backbone = get_backbone(
        conf.model.vit_backbone.arch,
        n_cls_tokens=2,
        masked_im_modeling=conf.model.masked_im_modeling,
        **conf.model.vit_backbone.get("kwargs", {}),
    )
    teacher_backbone = get_backbone(
        conf.model.vit_backbone.arch,
        n_cls_tokens=2,
        **conf.model.vit_backbone.get("kwargs", {}),
    )
    vit_embed_dim = student_backbone.embed_dim

    student_cnn_backbone = get_backbone(
        conf.model.cnn_backbone.arch, **conf.model.cnn_backbone.kwargs
    )
    teacher_cnn_backbone = get_backbone(
        conf.model.cnn_backbone.arch, **conf.model.cnn_backbone.kwargs
    )
    cnn_embed_dim = student_cnn_backbone.embed_dim

    head_kw = dict(
        out_dim=conf.model.n_prototypes,
        act="gelu",
        norm_last_layer=conf.model.norm_last_layer,
    )

    student = MultiCropWrapper(
        student_backbone,
        DINOHead(
            vit_embed_dim,
            **head_kw,
        ),
    )
    teacher = MultiCropWrapper(
        teacher_backbone,
        DINOHead(
            vit_embed_dim,
            **head_kw,
        ),
    )

    cnn_student = MultiCropWrapper(
        student_cnn_backbone,
        DINOHead(
            cnn_embed_dim,
            **head_kw,
        ),
    )

    cnn_teacher = MultiCropWrapper(
        teacher_cnn_backbone,
        DINOHead(
            cnn_embed_dim,
            **head_kw,
        ),
    )

    if conf.model.get("predictor", None) is not None:
        logging.info("Adding vit-to-cnn predictor")
        if conf.cross_distillation_target == "cnn_backbone":
            predictor_target_embed_dim = cnn_embed_dim
        elif conf.cross_distillation_target == "cnn_head":
            predictor_target_embed_dim = head_kw['out_dim']
        else: 
            raise ValueError(f"Unknown cross distillation target {conf.cross_distillation_target}")
        vit_to_cnn_predictor = TransformerPredictor(vit_embed_dim, predictor_target_embed_dim)
    else:
        vit_to_cnn_predictor = None

    return student, teacher, cnn_student, cnn_teacher, vit_to_cnn_predictor


def get_common_parameters(model1, model2):

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in model1.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in chain(model2.named_parameters()):
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))

    params_q = [
        param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common
    ]
    params_k = [
        param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common
    ]
    return params_q, params_k


def setup_for_ddp(student, teacher):
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[dist.get_rank()], broadcast_buffers=False
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[dist.get_rank()], broadcast_buffers=False
    )

    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher, teacher_without_ddp


class MomentumUpdater:
    def __init__(self, student, teacher, momentum_schedule):

        self.momentum_schedule = momentum_schedule

        # common params
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in student.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in teacher.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        assert (
            len(names_common) > 0
        ), "No common parameters found between student and teacher - check model architectures."
        logging.info(
            f"Found {len(names_common)} common parameters between student and teacher."
        )

        params_q = [
            param_q
            for name_q, param_q in zip(names_q, params_q)
            if name_q in names_common
        ]
        params_k = [
            param_k
            for name_k, param_k in zip(names_k, params_k)
            if name_k in names_common
        ]
        self.params_q = params_q
        self.params_k = params_k

    def update(self, it):
        # EMA update for the teacher
        breakpoint()

        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.params_k, self.params_q):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        breakpoint()


def update_optimizer(iter, optimizer, lr_schedulers, wd_schedulers):
    # update weight decay and learning rate according to their schedule
    lrs_and_wds_for_logging = {}
    for i, (param_group, lr_schedule, wd_schedule) in enumerate(
        zip(optimizer.param_groups, lr_schedulers, wd_schedulers)
    ):
        param_group["lr"] = lr_schedule[iter]
        param_group["weight_decay"] = wd_schedule[iter]
        lrs_and_wds_for_logging.update(
            {
                f"lr_{i}": lr_schedule[iter],
                f"wd_{i}": wd_schedule[iter],
            }
        )
    return lrs_and_wds_for_logging


def get_model_from_checkpoint(checkpoint_path, model="teacher"):

    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    conf = state_dict["args"]

    def extract_state_dict_with_prefix(state_dict, prefix):
        return {
            k.replace(prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

    vit_student, vit_teacher, cnn_student, cnn_teacher = get_models(conf)
    match model:
        case "student" | "vit_student":
            model = vit_student.backbone
            msg = model.load_state_dict(
                extract_state_dict_with_prefix(
                    state_dict["student"], "vit.module.backbone."
                )
            )
            print(msg)
        case "teacher" | "vit_teacher":
            model = vit_teacher.backbone
            msg = model.load_state_dict(
                extract_state_dict_with_prefix(state_dict["teacher"], "vit.backbone.")
            )
            print(msg)
        case "cnn_student":
            model = cnn_student.backbone
            msg = model.load_state_dict(
                extract_state_dict_with_prefix(
                    state_dict["student"], "cnn.module.backbone."
                )
            )
            print(msg)
        case "cnn_teacher":
            model = cnn_teacher.backbone
            msg = model.load_state_dict(
                extract_state_dict_with_prefix(state_dict["teacher"], "cnn.backbone.")
            )
            print(msg)
        case _:
            raise ValueError(f"Unknown model {model}")

    return model


class TransformerPredictor(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 384)
        self.transformer = TransformerEncoder(d_model=384, dropout=0, d_feed_forward=384*4, n_layers=6)
        self.output_proj = nn.Linear(384, output_dim)
        self.clstoken = nn.Parameter(torch.zeros(1, 1, 384))

    def forward(self, x: dict):

        # x = x['backbone_output'] # N x D, could have selected "head output" too

        x = self.input_proj(x)
        x = torch.cat(
            [self.clstoken.expand(x.shape[0], -1, -1), x], dim=1
        )  # prepend class token

        x = self.transformer(x)
        x = self.output_proj(x)
        return x[:, 0, :] # return class token output


def get_arg_parser():
    parser = argparse.ArgumentParser(
        "IBOT training with CNN distillation",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=["conf/main_ibot_multimodel/default.yaml"],
    )
    parser.add_argument(
        "--wandb_path", type=str, default=None, help="Path to wandb run to resume."
    )
    parser.add_argument(
        "--resume_wandb",
        action="store_true",
        help="If specified, tries to resume the wandb run.",
    )
    parser.add_argument(
        "--print_cfg", action="store_true", help="Print config and exit."
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Overrides for the config.",
    )
    return parser


def main():
    # conf = get_conf()
    parser = get_arg_parser()
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    if args.overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.overrides))
    if args.wandb_path is not None:
        # resume from wandb
        api = wandb.Api()
        run = api.run(args.wandb_path)
        run_conf = OmegaConf.create(run.config)
        print(f"{run.id} loaded from wandb.")
        # copy config
        conf = OmegaConf.merge(conf, run_conf)
        if args.resume_wandb:
            conf.load_from = conf.checkpoint_dir
            conf.wandb.id = run.id

    if args.print_cfg:
        rich.print(OmegaConf.to_yaml(conf))
        sys.exit(0)

    train_ibot(conf)

    # train_ibot(conf)


if __name__ == "__main__":
    main()