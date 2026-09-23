from medAI.utils import dino_utils as utils
import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from medAI.utils.distributed import concat_all_gather


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    ibot_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [
        param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common
    ]
    params_k = [
        param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common
    ]

    pred_labels, real_labels = [], []
    for it, (images, labels, masks) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output = teacher(images[: args.global_crops_number])
            student_output = student(
                images[: args.global_crops_number],
                mask=masks[: args.global_crops_number],
            )

            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = (
                student(images[args.global_crops_number :])[0]
                if len(images) > args.global_crops_number
                else None
            )
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(
                student_output, teacher_output, student_local_cls, masks, epoch
            )
            loss = all_loss.pop("loss")

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = concat_all_gather(probs1[0].max(dim=1)[1])
        pred2 = concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(concat_all_gather(labels.to(pred1.device)))

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return return_dict
