"""Reload a trained run and evaluate it on any split.

Examples:
    python evaluate.py --run runs/ref_s42 --split test
    python evaluate.py --run runs/ref_s42 --split val --drop-null
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import WindowDataset, make_loaders
from losses import build_loss
from models import build_model
from utils import (
    format_metrics,
    get_device,
    load_json,
    save_json,
    set_seed,
    write_per_class_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run directory, e.g. runs/ref_s42")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--drop-null", action="store_true", default=None)
    p.add_argument("--save-preds", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    config = load_json(run_dir / "config.json")
    split_name = config["split"]
    class_names = load_json(Path(args.data) / split_name / "class_names.json")
    set_seed(config["seed"])

    stride = args.stride
    if stride is None:
        stride = 1 if args.split == "test" else config["stride"]
    drop_null = config.get("drop_null", False) if args.drop_null is None else args.drop_null

    part_dir = Path(args.data) / split_name / args.split
    dataset = WindowDataset(
        part_dir,
        window=config["window"],
        stride=stride,
        augment="none",
        drop_null=drop_null,
        limit=args.limit,
        seed=config["seed"],
        training=False,
        class_names=class_names,
    )
    loaders = make_loaders(
        {"train": dataset, "val": dataset, "test": dataset},
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=config["seed"],
        test_batch_size=args.batch_size,
    )
    loader = loaders[args.split]

    device = get_device()
    model = build_model(
        model=config["model"],
        rnn=config["rnn"],
        n_conv=config["n_conv"],
        window=config["window"],
        dropout=config["dropout"],
        conv_filters=config["conv_filters"],
        kernel_size=config["kernel_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dense_hidden=config["dense_hidden"],
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = build_loss(config["loss"], config["focal_gamma"], config["focal_alpha"])

    from train import evaluate_loader

    loss, y_true, y_pred, probs = evaluate_loader(
        model, loader, criterion, device, class_names, collect_probs=args.save_preds
    )
    from utils import compute_metrics

    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["loss"] = float(loss)
    metrics["split"] = args.split
    metrics["stride"] = stride
    metrics["drop_null"] = bool(drop_null)
    metrics["run_id"] = run_dir.name
    print(f"[{args.split}] {format_metrics(metrics)} loss={loss:.4f} n={metrics['n_samples']}")

    save_json(run_dir / f"eval_{args.split}.json", metrics)
    write_per_class_csv(run_dir / f"per_class_{args.split}.csv", metrics, run_dir.name)
    np.save(run_dir / f"confusion_{args.split}.npy", np.asarray(metrics["confusion"]))
    if args.save_preds:
        np.savez_compressed(
            run_dir / f"preds_{args.split}.npz",
            y_true=y_true.astype(np.int16),
            y_pred=y_pred.astype(np.int16),
        )


if __name__ == "__main__":
    main()
