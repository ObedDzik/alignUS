import argparse
from collections import defaultdict
import json
import logging
import os
from argparse import ArgumentParser, Namespace
import time

import PIL
import hydra
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import rich_argparse
import torch
from PIL import Image
import medAI

import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

from medAI.layers.masked_prediction_module import get_bags_of_predictions
from medAI.datasets.nct2013 import data_accessor
from medAI.modeling import list_models, create_model
from medAI.losses.prostnfound import MaskedPredictionModule
from medAI.engine.prostnfound.evaluator import show_heatmap_prediction
from medAI.engine.prostnfound.prostnfound_model_wrapper import ProstNFoundModelInterface, ProstNFoundModelWrapper
from medAI.engine.prostnfound.evaluator import (
    ProstNFoundEvaluator as Evaluator,
)

# from src.loaders import get_dataloaders


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    else:
        state = None

    train_args = Namespace(**state["args"])

    if args.model is None:
        args.model = train_args.model
    if args.model_kw is None:
        args.model_kw = train_args.model_kw

    if args.save_checkpoint:
        torch.save(state, os.path.join(args.output_dir, "checkpoint.pth"))
    OmegaConf.save(
        state["args"],
        os.path.join(args.output_dir, "train_args.yaml"),
    )
    OmegaConf.save(
        args,
        os.path.join(args.output_dir, "test_args.yaml"),
    )

    model = ProstNFoundModelWrapper(create_model(args.model, **args.model_kw))
    print(model.load_state_dict(state["model"], strict=False))
    model.to(args.device)
    model.eval()
    if args.torch_compile:
        model = torch.compile(model)

    if args.get("data"):
        loaders = get_dataloaders(args.data, mode="test")
    elif "data" in vars(train_args):
        loaders = get_dataloaders(train_args.data, mode="test")
    else:
        loaders = get_dataloaders(train_args, mode="test")

    # maybe calibrate the temperature and bias of the model
    if args.calibration_mode == "pixel":
        do_calibration_pixel_wise_balanced_bce(
            model, loaders, args.calibrate_bias, args.calibrate_temperature
        )
    elif args.calibration_mode == "bag":
        do_calibration_bag_wise(
            model, loaders, args.calibrate_bias, args.calibrate_temperature
        )

    evaluator = Evaluator(
        log_images=False, include_patient_metrics=args.include_patient_metrics
    )
    accumulator = defaultdict(list)

    loader = loaders[args.split]

    # warmup
    for _ in range(10):
        batch = next(iter(loader))
        model(batch)

    for i, data in enumerate(tqdm(loader)):

        # measure inference
        t0 = time.perf_counter()

        with torch.amp.autocast_mode.autocast(
            device_type=torch.device(args.device).type, enabled=args.use_amp
        ):
            with torch.inference_mode():
                data = model(data)

        if args.postprocess:
            cancer_logits = data.pop("cancer_logits")
            heatmap = cancer_logits[0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample

            import skimage

            # import cv2
            #
            # blurred = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.5)
            # upsampled = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_LINEAR)

            blurred = skimage.filters.gaussian(heatmap, sigma=1.5)
            upsampled = skimage.transform.resize(
                blurred, (256, 256), order=1, anti_aliasing=True
            )
            upsampled = (upsampled * 255).astype(np.uint8)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]
        else:
            # get raw heatmap and also save as png
            heatmap = data["cancer_logits"][0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = Image.fromarray(heatmap)

        if args.device == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0
        accumulator["infer_time"].append(infer_time)

        if args.save_raw_heatmaps:
            # get raw heatmap and also save as png
            heatmap = Image.fromarray(heatmap)
            os.makedirs(os.path.join(args.output_dir, "raw_heatmaps"), exist_ok=True)
            heatmap.save(
                os.path.join(
                    args.output_dir, "raw_heatmaps", data["core_id"][0] + ".png"
                )
            )

        if args.save_rendered_heatmaps:

            patient_id = data["patient_id"][0]
            core_id = data["core_id"][0]

            output_file = os.path.join(
                args.output_dir, "heatmaps", patient_id, f"{core_id}.{args.save_format}"
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            show_heatmap_prediction(data)
            plt.savefig(
                output_file,
                format=args.save_format,
            )
            plt.close()

        evaluator(data)

    table = evaluator.accumulator.compute()
    table.to_csv(os.path.join(args.output_dir, "metrics_by_core.csv"))

    metrics = evaluator.aggregate_metrics()
    metrics["infer_time"] = np.array(accumulator["infer_time"]).mean()
    metrics = {k: float(v) for k, v in metrics.items()}

    print(json.dumps(metrics, indent=4))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def test(
    model: ProstNFoundModelInterface,
    loaders: dict,
    device="cuda",
    torch_compile=False,
    calibration_mode=None,
    calibrate_bias=True,
    calibrate_temperature=True,
    include_patient_metrics=False,
    split='test',
    use_amp=False,
    postprocess=False,
    save_raw_heatmaps=False,
    save_rendered_heatmaps=False,
    save_format='png',
    output_dir="./output",
):

    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    model.eval()
    if torch_compile:
        logging.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # maybe calibrate the temperature and bias of the model
    if calibration_mode == "pixel":
        do_calibration_pixel_wise_balanced_bce(
            model, loaders, calibrate_bias, calibrate_temperature
        )
    elif calibration_mode == "bag":
        do_calibration_bag_wise(model, loaders, calibrate_bias, calibrate_temperature)
    else:
        logging.info("No calibration performed.")

    evaluator = Evaluator(
        log_images=False, include_patient_metrics=include_patient_metrics
    )
    accumulator = defaultdict(list)

    loader = loaders[split]

    # warmup
    for _ in range(10):
        batch = next(iter(loader))
        model(batch)

    for i, data in enumerate(tqdm(loader)):

        # measure inference
        t0 = time.perf_counter()

        with torch.amp.autocast_mode.autocast(
            device_type=torch.device(device).type, enabled=use_amp
        ):
            with torch.inference_mode():
                data = model(data)

        if postprocess:
            cancer_logits = data.pop("cancer_logits")
            heatmap = cancer_logits[0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample

            import skimage

            blurred = skimage.filters.gaussian(heatmap, sigma=1.5)
            upsampled = skimage.transform.resize(
                blurred, (256, 256), order=1, anti_aliasing=True
            )
            upsampled = (upsampled * 255).astype(np.uint8)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]
        else:
            # get raw heatmap and also save as png
            heatmap = data["cancer_logits"][0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = Image.fromarray(heatmap)

        if device == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0
        accumulator["infer_time"].append(infer_time)

        if save_raw_heatmaps:
            # get raw heatmap and also save as png
            heatmap = Image.fromarray(heatmap)
            os.makedirs(os.path.join(output_dir, "raw_heatmaps"), exist_ok=True)
            heatmap.save(
                os.path.join(
                    output_dir, "raw_heatmaps", data["core_id"][0] + ".png"
                )
            )

        if save_rendered_heatmaps:

            patient_id = data["patient_id"][0]
            core_id = data["core_id"][0]

            output_file = os.path.join(
                output_dir, "heatmaps", patient_id, f"{core_id}.{save_format}"
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            show_heatmap_prediction(data)
            plt.savefig(
                output_file,
                format=save_format,
            )
            plt.close()

        evaluator(data)

    table = evaluator.accumulator.compute()
    table.to_csv(os.path.join(output_dir, "metrics_by_core.csv"))

    metrics = evaluator.aggregate_metrics()
    metrics["infer_time"] = np.array(accumulator["infer_time"]).mean()
    metrics = {k: float(v) for k, v in metrics.items()}

    print(json.dumps(metrics, indent=4))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def do_calibration_pixel_wise_balanced_bce(
    model,
    loaders,
    calibrate_bias=True,
    calibrate_temperature=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # extract all pixel predictions from val loader
    pixel_preds, pixel_labels, core_ids = extract_all_pixel_predictions(
        model, loaders["val"], device
    )
    core_ids = np.array(core_ids)

    # fit temperature and bias to center and scale the predictions
    temp = nn.Parameter(torch.ones(1))
    bias = nn.Parameter(torch.zeros(1))

    from torch.optim import LBFGS

    params = []
    if calibrate_bias:
        params.append(bias)
    if calibrate_temperature:
        params.append(temp)

    optim = LBFGS(params, lr=1e-3, max_iter=100, line_search_fn="strong_wolfe")

    # weight the loss to account for class imbalance
    pos_weight = (1 - pixel_labels).sum() / pixel_labels.sum()
    # encourage sensitivity over specificity
    pos_weight *= 1.6

    def closure():
        optim.zero_grad()
        logits = pixel_preds / temp + bias
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits[:, 0], pixel_labels)
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    model.temperature.data.copy_(temp)
    model.bias.data.copy_(bias)


def do_calibration_bag_wise(
    model,
    loaders,
    calibrate_bias=True,
    calibrate_temperature=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    bags_of_logits, involvement, label = extract_all_bag_predictions(
        model, loaders["val"], device
    )

    # fit temperature and bias to center and scale the predictions
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    bias = nn.Parameter(torch.zeros(1, device=device))

    from torch.optim import LBFGS

    pos_weight = (1 - label).sum() / label.sum()

    params = []
    if calibrate_bias:
        params.append(bias)
    if calibrate_temperature:
        params.append(log_temp)

    optim = LBFGS(params, lr=1e-1, max_iter=100)

    def closure():
        optim.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for bag_i, involvement_i, label_i in zip(bags_of_logits, involvement, label):
            bag_i = bag_i / log_temp.exp() + bias
            bag_i = bag_i.sigmoid()
            bag_i_mean = bag_i.mean()
            loss_i = (
                -involvement_i * bag_i_mean.log()
                - (1 - involvement_i) * (1 - bag_i_mean).log()
            )
            if label_i:
                loss_i = loss_i * pos_weight
            loss = loss + loss_i
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    model.temperature.data.copy_(log_temp.exp())
    model.bias.data.copy_(bias)


@torch.no_grad()
def extract_all_bag_predictions(model, loader, device):

    bags_of_logits = []
    involvement = []
    label = []

    for data in tqdm(loader, f"Running model..."):
        data = model(data)
        bags_of_logits.extend(
            get_bags_of_predictions(
                data["cancer_logits"], data["prostate_mask"], data["needle_mask"]
            )
        )
        involvement.append(data["involvement"].to(device))
        label.append(data["label"].to(device))

    involvement = torch.cat(involvement)
    label = torch.cat(label)

    return bags_of_logits, involvement, label


@torch.no_grad()
def extract_heatmap_and_data(model, batch, device):
    bmode = batch.pop("bmode").to(device)
    needle_mask = batch.pop("needle_mask").to(device)
    prostate_mask = batch.pop("prostate_mask").to(device)

    psa = batch["psa"].to(device)
    age = batch["age"].to(device)
    label = batch["label"].to(device)
    family_history = batch["family_history"].to(device)
    anatomical_location = batch["loc"].to(device)

    core_id = batch["core_id"][0]

    B = len(bmode)
    task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

    heatmap_logits = model(
        bmode,
        task_id=task_id,
        anatomical_location=anatomical_location,
        psa=psa,
        age=age,
        family_history=family_history,
        prostate_mask=prostate_mask,
        needle_mask=needle_mask,
    ).cpu()

    heatmap_logits = heatmap_logits[0, 0].sigmoid().cpu().numpy()
    bmode = bmode[0, 0].cpu().numpy()
    prostate_mask = prostate_mask[0, 0].cpu().numpy()
    needle_mask = needle_mask[0, 0].cpu().numpy()
    core_id = core_id

    return heatmap_logits, bmode, prostate_mask, needle_mask, core_id


def extract_all_pixel_predictions(model, loader, device):
    pixel_labels = []
    pixel_preds = []
    core_ids = []

    model.eval()
    model.to(device)

    for i, data in enumerate(tqdm(loader)):
        with torch.no_grad():
            data = model(data)

            prostate_mask = data["prostate_mask"].to(device)
            needle_mask = data["needle_mask"].to(device)
            heatmap_logits = data["cancer_logits"]
            label = data["label"]
            core_id = data["core_id"]

            # compute predictions
            masks = (prostate_mask > 0.5) & (needle_mask > 0.5)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)

            labels = torch.zeros(len(predictions), device=predictions.device)
            for i in range(len(predictions)):
                labels[i] = label[batch_idx[i]]
            pixel_preds.append(predictions.cpu())
            pixel_labels.append(labels.cpu())

            core_ids.extend(core_id[batch_idx[i]] for i in range(len(predictions)))

    pixel_preds = torch.cat(pixel_preds)
    pixel_labels = torch.cat(pixel_labels)

    return pixel_preds, pixel_labels, core_ids


def get_core_predictions_from_pixel_predictions(pixel_preds, pixel_labels, core_ids):
    data = []
    for core in np.unique(core_ids):
        mask = core_ids == core
        core_pred = pixel_preds[mask].sigmoid().mean().item()
        core_label = pixel_labels[mask][0].item()
        data.append({"core_id": core, "core_pred": core_pred, "core_label": core_label})

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    main()
