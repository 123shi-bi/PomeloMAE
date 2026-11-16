import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from timm.models.layers import trunc_normal_
from torch.utils.data import DataLoader, Dataset

from . import models_vit
from util.datasets import build_transform
from util.datasets_cmr import iter_fruit_views
import util.lr_decay as lrd
import util.lr_sched as lr_sched


class SSCImageDataset(Dataset):
    def __init__(self, records, transform, target_mean, target_std):
        self.records = records
        self.transform = transform
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = self.transform(image.convert("RGB"))
        target = (record["ssc"] - self.target_mean) / self.target_std
        return image, torch.tensor(target, dtype=torch.float32), index


def parse_args():
    parser = argparse.ArgumentParser("Fruit-view SSC regression fine-tuning")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument(
        "--finetune", type=Path, default=None,
        help="Optional pretrained checkpoint. Omit for supervised training "
             "from random ViT initialization.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-label", default="pomelomae_threefc_ccc02",
        choices=(
            "mae_linear_ccc", "mae_threefc_huber", "mae_threefc_ccc",
            "mae_threefc_ccc02", "mae_threefc_ccc03",
            "mae_baseline_threefc_ccc02", "pomelomae_threefc_ccc02",
            "hpm_threefc_ccc01", "hpm_threefc_ccc02",
            "vit_scratch_threefc_ccc02",
            "mae_fourfc_ccc", "mae_fivefc_ccc"))
    parser.add_argument("--model", default="vit_base_patch16_threefc_regression",
                        choices=(
                            "vit_base_patch16_linear_regression",
                            "vit_base_patch16_threefc_regression",
                            "vit_base_patch16_fourfc_regression",
                            "vit_base_patch16_fivefc_regression"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Peak learning rate (default: 5e-4).")
    parser.add_argument(
        "--blr", type=float, default=None,
        help="If --lr is omitted, scale this base rate by batch size / 256.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay", type=float, default=0.65)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--ccc-weight", type=float, default=0.2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--color-jitter", type=float, default=None)
    parser.add_argument("--aa", default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_measurements(path):
    frame = pd.read_csv(path, dtype=str)
    if {"fruit_id", "ssc"}.issubset(frame.columns):
        id_column, ssc_column = "fruit_id", "ssc"
    elif len(frame.columns) >= 12:
        id_column, ssc_column = frame.columns[0], frame.columns[11]
    else:
        raise RuntimeError("The measurement CSV must contain fruit_id and ssc columns.")
    if frame[id_column].isna().any():
        raise RuntimeError("Measurement file contains missing fruit IDs.")
    if frame[id_column].duplicated().any():
        raise RuntimeError("Measurement file contains duplicate fruit IDs.")
    lookup = {
        str(row[id_column]): float(row[ssc_column])
        for _, row in frame.iterrows()
    }
    if not all(math.isfinite(value) for value in lookup.values()):
        raise RuntimeError("Measurement file contains non-finite SSC values.")
    return lookup, str(id_column), str(ssc_column)


def collect_records(data_path, split, measurements):
    records = []
    for class_name, fruit_id, images in iter_fruit_views(data_path / split):
        if not fruit_id.isdecimal():
            raise ValueError(f"Expected a numeric fruit ID in {split}, found {fruit_id!r}")
        if fruit_id not in measurements:
            raise RuntimeError(f"Missing SSC value for fruit {fruit_id}")
        for view_index, path in enumerate(images):
            records.append({
                "split": split,
                "class": class_name,
                "fruit_id": fruit_id,
                "view_index": view_index,
                "path": path,
                "ssc": float(measurements[fruit_id]),
            })
    return records


def load_pretrained_encoder(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = dict(checkpoint["model"])
    if hasattr(model, "prepare_pretrained_state_dict"):
        checkpoint_model = model.prepare_pretrained_state_dict(
            checkpoint_model)
    model_state = model.state_dict()
    for key in list(checkpoint_model):
        if (key.startswith("head.")
                and (key not in model_state
                     or checkpoint_model[key].shape != model_state[key].shape)):
            del checkpoint_model[key]
    message = model.load_state_dict(checkpoint_model, strict=False)
    expected_missing = {
        key for key in model_state
        if (key.startswith("head.") or key.startswith("fc_norm."))
        and not key.endswith("num_batches_tracked")
        and key not in checkpoint_model
    }
    if hasattr(model, "expected_missing_pretrained_keys"):
        expected_missing.update(model.expected_missing_pretrained_keys())
    if set(message.missing_keys) != expected_missing:
        raise RuntimeError(
            f"Unexpected missing keys while loading encoder: {message.missing_keys}")
    encoder_keys = [
        key for key, value in checkpoint_model.items()
        if key in model_state and model_state[key].shape == value.shape
        and not key.startswith("head.")
    ]
    if not encoder_keys:
        raise RuntimeError("No encoder tensors were loaded from the checkpoint.")
    loaded_state = model.state_dict()
    mismatched = [
        key for key in encoder_keys
        if not torch.equal(loaded_state[key].cpu(), checkpoint_model[key].cpu())
    ]
    if mismatched:
        raise RuntimeError(f"Loaded encoder tensors changed: {mismatched[:10]}")
    if hasattr(model, "reset_regression_head"):
        model.reset_regression_head()
    elif isinstance(model.head, torch.nn.Linear):
        trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            torch.nn.init.zeros_(model.head.bias)
    else:
        raise RuntimeError(
            "Nonlinear regression heads must implement reset_regression_head().")
    return checkpoint, message, len(encoder_keys)


class HuberCccLoss(torch.nn.Module):
    def __init__(self, delta=1.0, ccc_weight=0.2, eps=1e-8):
        super().__init__()
        if ccc_weight < 0:
            raise ValueError("ccc_weight must be non-negative.")
        self.huber = torch.nn.HuberLoss(delta=float(delta))
        self.ccc_weight = float(ccc_weight)
        self.eps = float(eps)

    def forward(self, predictions, targets):
        huber = self.huber(predictions, targets)
        if self.ccc_weight == 0:
            return huber
        pred_centered = predictions - predictions.mean()
        target_centered = targets - targets.mean()
        pred_var = pred_centered.square().mean()
        target_var = target_centered.square().mean()
        covariance = (pred_centered * target_centered).mean()
        mean_difference = predictions.mean() - targets.mean()
        ccc = (
            2.0 * covariance
            / (
                pred_var + target_var + mean_difference.square()
                + self.eps
            )
        )
        return huber + self.ccc_weight * (1.0 - ccc)


def regression_metrics(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson_r = float("nan")
        spearman_rho = float("nan")
    else:
        pearson_r = float(pearsonr(y_true, y_pred).statistic)
        spearman_rho = float(spearmanr(y_true, y_pred).statistic)
    true_mean = float(np.mean(y_true))
    pred_mean = float(np.mean(y_pred))
    covariance = float(np.mean(
        (y_true - true_mean) * (y_pred - pred_mean)))
    ccc = (
        2.0 * covariance
        / (
            float(np.var(y_true)) + float(np.var(y_pred))
            + (true_mean - pred_mean) ** 2 + 1e-12
        )
    )
    return {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "ccc": float(ccc),
    }


@torch.no_grad()
def evaluate(model, loader, records, device, target_mean, target_std):
    model.eval()
    predictions = np.empty(len(records), dtype=np.float64)
    losses = []
    criterion = torch.nn.HuberLoss(delta=1.0)
    for images, targets, indices in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
        losses.append((float(loss.item()), len(indices)))
        raw_predictions = outputs.float().cpu().numpy() * target_std + target_mean
        predictions[indices.numpy()] = raw_predictions

    image_truth = np.array([record["ssc"] for record in records], dtype=np.float64)
    image_result = regression_metrics(image_truth, predictions)
    image_result["huber_loss_standardized"] = float(
        sum(value * count for value, count in losses)
        / sum(count for _, count in losses))

    frame = pd.DataFrame({
        "fruit_id": [record["fruit_id"] for record in records],
        "class": [record["class"] for record in records],
        "view_index": [record["view_index"] for record in records],
        "image_path": [str(record["path"]) for record in records],
        "true_ssc": image_truth,
        "pred_ssc": predictions,
    })
    fruit_frame = frame.groupby(["fruit_id", "class"], as_index=False).agg(
        true_ssc=("true_ssc", "first"),
        pred_ssc=("pred_ssc", "mean"),
        view_pred_std=("pred_ssc", lambda value: float(np.std(value, ddof=0))),
        n_views=("pred_ssc", "size"),
    )
    if not (fruit_frame["n_views"] == 4).all():
        raise RuntimeError("Validation aggregation did not contain four views per fruit.")
    fruit_result = regression_metrics(
        fruit_frame["true_ssc"].to_numpy(),
        fruit_frame["pred_ssc"].to_numpy())
    fruit_result["mean_four_view_prediction_std"] = float(
        fruit_frame["view_pred_std"].mean())
    return {
        "image_level": image_result,
        "fruit_level": fruit_result,
    }, frame, fruit_frame


def train_one_epoch(
        model, loader, optimizer, scaler, criterion, device, epoch, args):
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets, _) in enumerate(loader):
        if args.smoke_batches and step >= args.smoke_batches:
            break
        progress = epoch + step / max(1, len(loader))
        lr_sched.adjust_learning_rate(optimizer, progress, args)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
    return {
        "loss": float(np.mean(losses)),
        "lr": float(max(group["lr"] for group in optimizer.param_groups)),
        "batches": int(len(losses)),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    measurements, id_column, ssc_column = load_measurements(args.measurements)
    records = {
        split: collect_records(args.data_path, split, measurements)
        for split in ("train", "val")
    }
    train_ids = {record["fruit_id"] for record in records["train"]}
    val_ids = {record["fruit_id"] for record in records["val"]}
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise RuntimeError(f"Fruit-level train/validation overlap: {sorted(overlap)}")
    if len(val_ids) < 2:
        raise ValueError("The validation split needs at least two fruits for regression metrics.")
    train_fruit_ssc = np.array(
        [measurements[fruit_id] for fruit_id in sorted(train_ids, key=int)],
        dtype=np.float64)
    target_mean = float(train_fruit_ssc.mean())
    target_std = float(train_fruit_ssc.std(ddof=0))
    if target_std <= 0:
        raise RuntimeError("Training SSC standard deviation must be positive.")

    transform_args = argparse.Namespace(
        input_size=args.input_size,
        color_jitter=args.color_jitter,
        aa=args.aa,
        reprob=args.reprob,
        remode=args.remode,
        recount=args.recount,
    )
    train_dataset = SSCImageDataset(
        records["train"], build_transform(True, transform_args),
        target_mean, target_std)
    val_dataset = SSCImageDataset(
        records["val"], build_transform(False, transform_args),
        target_mean, target_std)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True, generator=generator)
    if len(train_loader) == 0:
        raise ValueError("The training split contains fewer images than one batch.")
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=False)

    effective_batch_size = int(args.batch_size)
    if args.lr is None:
        args.lr = 5e-4 if args.blr is None else args.blr * effective_batch_size / 256.0
    print(f"base lr: {args.lr * 256.0 / effective_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"effective batch size: {effective_batch_size}")

    if args.model not in models_vit.__dict__:
        raise RuntimeError(f"Unknown regression model: {args.model}")
    model = models_vit.__dict__[args.model](
        num_classes=1, drop_path_rate=args.drop_path, global_pool=True)
    if args.finetune is None:
        checkpoint = {}
        unexpected_pretraining_keys = []
        loaded_encoder_tensors = 0
    else:
        checkpoint, load_message, loaded_encoder_tensors = (
            load_pretrained_encoder(model, args.finetune))
        unexpected_pretraining_keys = list(load_message.unexpected_keys)
    model.to(device)

    parameter_groups = lrd.param_groups_lrd(
        model, args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = HuberCccLoss(
        delta=args.huber_delta, ccc_weight=args.ccc_weight)

    provenance = {
        "analysis": "single-view SSC end-to-end regression fine-tuning",
        "model_label": args.model_label,
        "model": args.model,
        "pretraining_checkpoint": (
            None if args.finetune is None else str(args.finetune)),
        "pretraining_epoch": checkpoint.get("epoch"),
        "loaded_encoder_tensors": loaded_encoder_tensors,
        "reinitialized_pretraining_keys": list(
            getattr(model, "_reinitialized_pretraining_keys", ())),
        "unexpected_pretraining_keys": unexpected_pretraining_keys,
        "data_path": str(args.data_path),
        "measurements": str(args.measurements),
        "measurement_id_column": id_column,
        "measurement_ssc_column": ssc_column,
        "train_fruits": len(train_ids),
        "validation_fruits": len(val_ids),
        "train_images": len(records["train"]),
        "validation_images": len(records["val"]),
        "train_val_fruit_overlap": len(overlap),
        "target_mean_train_fruits": target_mean,
        "target_std_train_fruits": target_std,
        "args": vars(args) | {
            "data_path": str(args.data_path),
            "measurements": str(args.measurements),
            "finetune": (
                None if args.finetune is None else str(args.finetune)),
            "output_dir": str(args.output_dir),
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    best_mae = float("inf")
    best_epoch = -1
    best_r2 = float("-inf")
    best_r2_epoch = -1
    best_image_r2 = float("-inf")
    best_image_r2_epoch = -1
    log_path = args.output_dir / "log.txt"
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion,
            device, epoch, args)
        val_stats, image_frame, fruit_frame = evaluate(
            model, val_loader, records["val"], device,
            target_mean, target_std)
        row = {
            "epoch": epoch,
            "train": train_stats,
            "validation": val_stats,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

        fruit_mae = val_stats["fruit_level"]["mae"]
        if fruit_mae < best_mae:
            best_mae = fruit_mae
            best_epoch = epoch
            image_frame.to_csv(
                args.output_dir / "val_image_predictions_best.csv",
                index=False, encoding="utf-8-sig")
            fruit_frame.to_csv(
                args.output_dir / "val_fruit_predictions_best.csv",
                index=False, encoding="utf-8-sig")
            best_result = {
                "model_label": args.model_label,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "selection_metric": "validation fruit-level MAE",
                "validation": val_stats,
            }
            (args.output_dir / "best_metrics.json").write_text(
                json.dumps(best_result, ensure_ascii=False, indent=2),
                encoding="utf-8")

        fruit_r2 = val_stats["fruit_level"]["r2"]
        if fruit_r2 > best_r2:
            best_r2 = fruit_r2
            best_r2_epoch = epoch
            image_frame.to_csv(
                args.output_dir / "val_image_predictions_best_r2.csv",
                index=False, encoding="utf-8-sig")
            fruit_frame.to_csv(
                args.output_dir / "val_fruit_predictions_best_r2.csv",
                index=False, encoding="utf-8-sig")
            best_r2_result = {
                "model_label": args.model_label,
                "model": args.model,
                "seed": args.seed,
                "best_epoch": best_r2_epoch,
                "selection_metric": "validation fruit-level R2",
                "validation": val_stats,
            }
            (args.output_dir / "best_metrics_r2.json").write_text(
                json.dumps(best_r2_result, ensure_ascii=False, indent=2),
                encoding="utf-8")

        image_r2 = val_stats["image_level"]["r2"]
        if image_r2 > best_image_r2:
            best_image_r2 = image_r2
            best_image_r2_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "seed": args.seed,
                "target_mean": target_mean,
                "target_std": target_std,
                "validation": val_stats,
                "model_label": args.model_label,
                "model_name": args.model,
                "pretraining_checkpoint": (
                    None if args.finetune is None else str(args.finetune)),
                "selection_metric": "validation image-level R2",
            }, args.output_dir / "checkpoint-best-image-r2.pth")
            image_frame.to_csv(
                args.output_dir / "val_image_predictions_best_image_r2.csv",
                index=False, encoding="utf-8-sig")
            fruit_frame.to_csv(
                args.output_dir / "val_fruit_predictions_best_image_r2.csv",
                index=False, encoding="utf-8-sig")
            best_image_r2_result = {
                "model_label": args.model_label,
                "model": args.model,
                "seed": args.seed,
                "best_epoch": best_image_r2_epoch,
                "selection_metric": "validation image-level R2",
                "validation": val_stats,
            }
            (args.output_dir / "best_metrics_image_r2.json").write_text(
                json.dumps(
                    best_image_r2_result, ensure_ascii=False, indent=2),
                encoding="utf-8")

    print(json.dumps({
        "status": "complete",
        "model_label": args.model_label,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_fruit_mae": best_mae,
        "best_r2_epoch": best_r2_epoch,
        "best_fruit_r2": best_r2,
        "best_image_r2_epoch": best_image_r2_epoch,
        "best_image_r2": best_image_r2,
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
