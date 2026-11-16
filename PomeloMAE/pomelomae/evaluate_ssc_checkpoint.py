import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import models_vit
from .finetune_ssc_regression import (
    SSCImageDataset,
    collect_records,
    evaluate,
    load_measurements,
    seed_everything,
)
from util.datasets import build_transform


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate one validation-selected SSC regression checkpoint")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_rpd(metrics, truths):
    result = dict(metrics)
    sample_std = float(np.std(np.asarray(truths, dtype=np.float64), ddof=1))
    result["truth_sample_std"] = sample_std
    result["rpd"] = float(sample_std / result["rmse"]) if result["rmse"] > 0 else None
    return result


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    measurements, id_column, ssc_column = load_measurements(args.measurements)
    records = {
        split: collect_records(args.data_path, split, measurements)
        for split in ("train", "val", "test")
    }
    fruit_ids = {
        split: {record["fruit_id"] for record in split_records}
        for split, split_records in records.items()
    }
    overlaps = {
        "train_val": sorted(fruit_ids["train"] & fruit_ids["val"]),
        "train_test": sorted(fruit_ids["train"] & fruit_ids["test"]),
        "val_test": sorted(fruit_ids["val"] & fruit_ids["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Fruit-level split overlap: {overlaps}")
    if len(fruit_ids[args.split]) < 2:
        raise ValueError(f"The {args.split} split needs at least two fruits for regression metrics.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("selection_metric") != "validation image-level R2":
        raise RuntimeError(
            "Checkpoint was not selected by validation image-level R2: "
            f"{checkpoint.get('selection_metric')!r}")
    model_name = args.model or checkpoint.get("model_name")
    if not model_name or model_name not in models_vit.__dict__:
        raise RuntimeError(f"Unknown or missing regression model: {model_name!r}")

    train_ssc = np.array(
        [measurements[fruit_id]
         for fruit_id in sorted(fruit_ids["train"], key=int)],
        dtype=np.float64,
    )
    expected_mean = float(train_ssc.mean())
    expected_std = float(train_ssc.std(ddof=0))
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])
    if not math.isclose(target_mean, expected_mean, abs_tol=1e-12):
        raise RuntimeError(
            f"Checkpoint target mean mismatch: {target_mean} != {expected_mean}")
    if not math.isclose(target_std, expected_std, abs_tol=1e-12):
        raise RuntimeError(
            f"Checkpoint target std mismatch: {target_std} != {expected_std}")

    transform_args = argparse.Namespace(
        input_size=args.input_size,
        color_jitter=None,
        aa="rand-m9-mstd0.5-inc1",
        reprob=0.25,
        remode="pixel",
        recount=1,
    )
    dataset = SSCImageDataset(
        records[args.split], build_transform(False, transform_args),
        target_mean, target_std)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=False)

    model = models_vit.__dict__[model_name](
        num_classes=1, drop_path_rate=0.0, global_pool=True)
    load_message = model.load_state_dict(checkpoint["model"], strict=True)
    if load_message.missing_keys or load_message.unexpected_keys:
        raise RuntimeError(f"Checkpoint load mismatch: {load_message}")
    model.to(device)

    metrics, image_frame, fruit_frame = evaluate(
        model, loader, records[args.split], device, target_mean, target_std)
    metrics["image_level"] = add_rpd(
        metrics["image_level"], image_frame["true_ssc"].to_numpy())
    metrics["fruit_level"] = add_rpd(
        metrics["fruit_level"], fruit_frame["true_ssc"].to_numpy())

    image_path = args.output_dir / f"{args.split}_image_predictions.csv"
    fruit_path = args.output_dir / f"{args.split}_fruit_predictions.csv"
    metrics_path = args.output_dir / f"{args.split}_metrics.json"
    provenance_path = args.output_dir / "evaluation_provenance.json"
    image_frame.to_csv(image_path, index=False, encoding="utf-8-sig")
    fruit_frame.to_csv(fruit_path, index=False, encoding="utf-8-sig")

    result = {
        "status": "complete",
        "split": args.split,
        "selection_protocol": (
            "checkpoint selected exclusively by validation image-level R2; "
            f"{args.split} evaluated once after selection"),
        "model_label": checkpoint.get("model_label"),
        "model": model_name,
        "seed": args.seed,
        "selected_validation_epoch": int(checkpoint["epoch"]),
        "selected_validation_metrics": checkpoint.get("validation"),
        "evaluation": metrics,
    }
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    provenance = {
        "analysis": "sealed split evaluation of validation-selected checkpoint",
        "data_path": str(args.data_path),
        "measurements": str(args.measurements),
        "measurement_id_column": id_column,
        "measurement_ssc_column": ssc_column,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "pretraining_checkpoint": checkpoint.get("pretraining_checkpoint"),
        "model_label": checkpoint.get("model_label"),
        "model": model_name,
        "split": args.split,
        "fruit_counts": {key: len(value) for key, value in fruit_ids.items()},
        "image_counts": {key: len(value) for key, value in records.items()},
        "split_overlaps": {key: len(value) for key, value in overlaps.items()},
        "target_mean_train_fruits": target_mean,
        "target_std_train_fruits": target_std,
        "args": vars(args) | {
            "data_path": str(args.data_path),
            "measurements": str(args.measurements),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
