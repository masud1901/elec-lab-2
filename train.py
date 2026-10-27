"""Train one DeepConvLSTM ablation run.

Examples
--------
Reference:
    python train.py --run-id ref_s42 --seed 42

Part A:
    python train.py --run-id A_cnn_s42 --model cnn --seed 42
    python train.py --run-id A_gru_s42 --rnn gru --seed 42

Part B:
    python train.py --run-id B_conv0_s42 --n-conv 0 --seed 42

Part C:
    python train.py --run-id C_w20_s42 --window 20 --seed 42

Part D:
    python train.py --run-id D_no_dropout_s42 --dropout 0 --seed 42

Part E:
    python train.py --run-id E_lr1e-4_s42 --lr 1e-4 --seed 42
    python train.py --run-id E_warmup_s42 --lr 1e-2 --warmup-epochs 5 --seed 42

Part F:
    python train.py --run-id F_focal_s42 --loss focal --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from data import make_datasets, make_loaders
from losses import build_loss
from models import build_model, count_parameters, init_weights
from utils import (
    AverageMeter,
    Logger,
    compute_metrics,
    format_metrics,
    get_device,
    git_info,
    human_time,
    load_json,
    save_json,
    set_seed,
    write_per_class_csv,
)


class DivergedError(RuntimeError):
    """Raised when a run produces a non-finite loss (e.g. a too-high LR)."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=None, help="directory name under --out")
    p.add_argument("--out", default="runs")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--split", default="paper", choices=["paper", "course"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--part", default="", help="ablation part label, e.g. A/B/C/D/E/F")

    # architecture
    p.add_argument("--model", default="deepconvlstm", choices=["deepconvlstm", "cnn"])
    p.add_argument("--rnn", default="lstm", choices=["lstm", "gru"])
    p.add_argument("--n-conv", type=int, default=4)
    p.add_argument("--conv-filters", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dense-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)

    # data / window
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--stride", type=int, default=None, help="train/val stride (default window//2)")
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--augment", default="full", choices=["full", "none", "jitter", "scale", "shift"])
    p.add_argument("--drop-null", action="store_true")

    # optimisation
    p.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "adam"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.9, help="RMSProp decay")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--micro-batch-size", type=int, default=0,
                   help="split each batch into micro-batches (gradient accumulation) to save memory")
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--clip-grad", type=float, default=0.0, help="0 = no clipping")
    p.add_argument("--select-metric", default="macro_f1", choices=["macro_f1", "weighted_f1"])
    p.add_argument("--loss", default="ce", choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=-1.0, help="<0 = no alpha")

    # misc
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="limit windows per split (debug)")
    p.add_argument("--no-test", action="store_true")
    p.add_argument("--save-probs", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue from best.pt if present")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--smoke", action="store_true", help="tiny fast run")
    return p.parse_args()


def default_run_id(args: argparse.Namespace) -> str:
    if args.part:
        base = f"{args.part}"
        bits = []
        if args.model != "deepconvlstm":
            bits.append(args.model)
        if args.rnn != "lstm":
            bits.append(args.rnn)
        if args.n_conv != 4:
            bits.append(f"conv{args.n_conv}")
        if args.window != 24:
            bits.append(f"w{args.window}")
        if args.dropout == 0:
            bits.append("nodrop")
        if args.weight_decay == 0:
            bits.append("nowd")
        if args.augment == "none":
            bits.append("noaug")
        if args.lr != 1e-3:
            bits.append(f"lr{args.lr:g}")
        if args.warmup_epochs:
            bits.append(f"warm{args.warmup_epochs}")
        if args.loss == "focal":
            bits.append("focal")
        base += "_" + "_".join(bits) if bits else "_ref"
        return f"{base}_s{args.seed}"
    return f"run_s{args.seed}"


def build_config(args: argparse.Namespace) -> Dict[str, object]:
    config = vars(args).copy()
    config["stride"] = args.stride if args.stride is not None else max(1, args.window // 2)
    config["command"] = " ".join([sys.executable] + sys.argv)
    config["git"] = git_info(".")
    config["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["environment"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    return config


def warmup_lambda(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / float(warmup_epochs))


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    class_names,
    collect_probs: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    loss_meter = AverageMeter("loss")
    y_true, y_pred, probs = [], [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), x.size(0))
        pred = logits.argmax(dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        if collect_probs:
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    probs_np = np.concatenate(probs) if collect_probs else None
    return loss_meter.avg, y_true_np, y_pred_np, probs_np


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float = 0.0,
    micro_batch_size: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train()
    loss_meter = AverageMeter("loss")
    y_true, y_pred = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        n = x.size(0)
        micro = micro_batch_size if micro_batch_size and micro_batch_size < n else n
        loss_sum = 0.0
        for start in range(0, n, micro):
            xb = x[start : start + micro]
            yb = y[start : start + micro]
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise DivergedError(f"non-finite loss: {loss.item()}")
            loss_sum += loss.item() * xb.size(0)
            # scale so that the accumulated gradient equals the full-batch gradient
            (loss * (xb.size(0) / n)).backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            y_true.append(yb.detach().cpu().numpy())
            y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
        optimizer.step()
        loss_meter.update(loss_sum / n, n)
    return loss_meter.avg, np.concatenate(y_true), np.concatenate(y_pred)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = 1
        args.limit = args.limit or 600
        args.batch_size = min(args.batch_size, 64)
        args.num_workers = 0
        args.no_test = False

    if args.run_id is None:
        args.run_id = default_run_id(args)
    run_dir = Path(args.out) / args.run_id
    if (
        run_dir.exists()
        and any(run_dir.iterdir())
        and not (args.overwrite or args.resume)
    ):
        raise SystemExit(f"{run_dir} already exists; use --overwrite or another --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(run_dir / "train.log")
    config = build_config(args)
    save_json(run_dir / "config.json", config)
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device()
    log(f"[run] {args.run_id}")
    log(f"[device] {device} {config['environment']['gpu'] or ''}")
    log(f"[config] {json.dumps({k: v for k, v in config.items() if k not in ('environment', 'command')}, sort_keys=True)}")

    class_names = load_json(Path(args.data) / args.split / "class_names.json")
    datasets = make_datasets(
        data_dir=args.data,
        split=args.split,
        window=args.window,
        train_stride=config["stride"],
        test_stride=args.test_stride,
        augment=args.augment,
        drop_null=args.drop_null,
        limit=args.limit,
        seed=args.seed,
        class_names=class_names,
    )
    for part, ds in datasets.items():
        log(f"[data:{part}] {ds.summary()}")

    loaders = make_loaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        test_batch_size=args.test_batch_size,
    )

    model = build_model(
        model=args.model,
        rnn=args.rnn,
        n_conv=args.n_conv,
        window=args.window,
        dropout=args.dropout,
        conv_filters=args.conv_filters,
        kernel_size=args.kernel_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dense_hidden=args.dense_hidden,
    ).to(device)
    model.apply(init_weights)
    n_params = count_parameters(model)
    log(f"[model] {args.model} rnn={args.rnn} n_conv={args.n_conv} window={args.window} "
        f"dropout={args.dropout} params={n_params:,}")

    criterion = build_loss(args.loss, args.focal_gamma, args.focal_alpha)
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            alpha=args.alpha,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    scheduler = None
    if args.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: warmup_lambda(e, args.warmup_epochs)
        )

    history_path = run_dir / "history.csv"
    history_fields = [
        "epoch", "lr", "train_loss", "train_acc", "train_macro_f1", "train_weighted_f1",
        "val_loss", "val_acc", "val_macro_f1", "val_weighted_f1", "val_gesture_macro_f1",
        "epoch_time_s",
    ]

    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    start_time = time.time()
    stopped_early = False
    diverged = False
    first_epoch = 1

    # Resume support: continue from the best checkpoint of an interrupted run.
    ckpt_path = run_dir / "best.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:  # noqa: BLE001
                log(f"[resume] could not restore optimizer state: {exc}")
        best_score = float(ckpt.get("val_score", -1.0))
        best_epoch = int(ckpt.get("epoch", 0))
        first_epoch = best_epoch + 1
        log(f"[resume] continuing from epoch {best_epoch} (val score {best_score:.4f})")
    if not history_path.exists() or first_epoch == 1:
        with history_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields).writeheader()

    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.time()

        try:
            train_loss, tr_true, tr_pred = train_one_epoch(
                model,
                loaders["train"],
                criterion,
                optimizer,
                device,
                clip_grad=args.clip_grad,
                micro_batch_size=args.micro_batch_size,
            )
        except DivergedError as exc:
            log(f"[diverge] epoch {epoch}: {exc}")
            diverged = True
            break
        train_metrics = compute_metrics(tr_true, tr_pred, class_names)
        val_loss, va_true, va_pred, _ = evaluate_loader(
            model, loaders["val"], criterion, device, class_names
        )
        if not np.isfinite(val_loss):
            log(f"[diverge] epoch {epoch}: non-finite validation loss")
            diverged = True
            break
        val_metrics = compute_metrics(va_true, va_pred, class_names)
        score = float(val_metrics[args.select_metric])
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "lr": f"{lr_now:.8g}",
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_metrics['accuracy']:.6f}",
            "train_macro_f1": f"{train_metrics['macro_f1']:.6f}",
            "train_weighted_f1": f"{train_metrics['weighted_f1']:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_metrics['accuracy']:.6f}",
            "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
            "val_weighted_f1": f"{val_metrics['weighted_f1']:.6f}",
            "val_gesture_macro_f1": f"{val_metrics['gesture_macro_f1']:.6f}",
            "epoch_time_s": f"{epoch_time:.2f}",
        }
        with history_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields).writerow(row)

        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_score": score,
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - start_time
        avg_epoch = elapsed / epoch
        eta = avg_epoch * (args.epochs - epoch)
        log(
            f"[epoch {epoch:3d}/{args.epochs}] lr={lr_now:.2e} "
            f"train loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} "
            f"macroF1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macroF1={val_metrics['macro_f1']:.4f} weightedF1={val_metrics['weighted_f1']:.4f} "
            f"| best={best_score:.4f}@{best_epoch} "
            f"| {epoch_time:.1f}s eta={human_time(eta)}"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            log(f"[early-stop] no improvement for {args.patience} epochs")
            stopped_early = True
            break

        if scheduler is not None:
            scheduler.step()

    total_time = time.time() - start_time
    log(f"[train] finished in {human_time(total_time)}; best epoch={best_epoch} "
        f"val {args.select_metric}={best_score:.4f}")

    if not (run_dir / "best.pt").exists():
        save_json(
            run_dir / "run.json",
            {
                "run_id": args.run_id,
                "part": args.part,
                "config": config,
                "param_count": int(n_params),
                "best_epoch": int(best_epoch),
                "epochs_run": int(epoch),
                "status": "diverged",
                "train_time_s": float(total_time),
                "val": None,
                "test": None,
            },
        )
        log("[done] run diverged before any finite validation epoch")
        return

    # Reload the best checkpoint and evaluate the untouched test split once.
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, va_true, va_pred, _ = evaluate_loader(
        model, loaders["val"], criterion, device, class_names
    )
    val_metrics = compute_metrics(va_true, va_pred, class_names)
    write_per_class_csv(run_dir / "per_class_val.csv", val_metrics, args.run_id)

    test_metrics = None
    if not args.no_test:
        test_start = time.time()
        test_loss, te_true, te_pred, te_probs = evaluate_loader(
            model,
            loaders["test"],
            criterion,
            device,
            class_names,
            collect_probs=args.save_probs,
        )
        test_metrics = compute_metrics(te_true, te_pred, class_names)
        test_metrics["loss"] = float(test_loss)
        test_metrics["eval_time_s"] = float(time.time() - test_start)
        write_per_class_csv(run_dir / "per_class.csv", test_metrics, args.run_id)
        np.save(run_dir / "confusion_test.npy", np.asarray(test_metrics["confusion"]))
        np.savez_compressed(
            run_dir / "test_predictions.npz",
            y_true=te_true.astype(np.int16),
            y_pred=te_pred.astype(np.int16),
        )
        if args.save_probs and te_probs is not None:
            np.savez_compressed(run_dir / "test_probs.npz", probs=te_probs.astype(np.float16))
        log(f"[test] {format_metrics(test_metrics)} loss={test_loss:.4f} "
            f"n={test_metrics['n_samples']}")

    result = {
        "run_id": args.run_id,
        "part": args.part,
        "config": config,
        "param_count": int(n_params),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "status": "diverged" if diverged else "ok",
        "stopped_early": bool(stopped_early),
        "train_time_s": float(total_time),
        "best_val_score": float(best_score),
        "select_metric": args.select_metric,
        "class_names": class_names,
        "val": val_metrics,
        "test": test_metrics,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(run_dir / "run.json", result)
    log(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
