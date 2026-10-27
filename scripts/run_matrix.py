"""Parallel runner for the ablation matrix.

Runs several trainings concurrently on one GPU.  A single DeepConvLSTM run
only uses ~1.4 GB of the 4 GB card and keeps the SMs at ~50% load, so two
light jobs roughly double throughput.  Jobs that need a lot of memory (the
dense CNN baseline, long windows) are scheduled alone.

Usage:
    python scripts/run_matrix.py --seeds 42 43 44 --jobs 2
    python scripts/run_matrix.py --seeds 42 --parts A B
    python scripts/run_matrix.py --seeds 42 --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
PY = os.environ.get("PY", str(ROOT / ".venv" / "bin" / "python"))

# (run-id, extra args, rough GPU memory requirement in MiB)
VARIANTS: List[Tuple[str, List[str], int]] = [
    ("ref", ["--part", "0"], 1500),
    ("A_cnn", ["--part", "A", "--model", "cnn"], 3600),
    ("A_gru", ["--part", "A", "--rnn", "gru"], 1300),
    ("B_conv0", ["--part", "B", "--n-conv", "0"], 500),
    ("B_conv1", ["--part", "B", "--n-conv", "1"], 1500),
    ("B_conv2", ["--part", "B", "--n-conv", "2"], 1500),
    ("C_w20", ["--part", "C", "--window", "20"], 1400),
    ("C_w42", ["--part", "C", "--window", "42"], 2500),
    ("C_w60", ["--part", "C", "--window", "60"], 3600),
    ("D_no_dropout", ["--part", "D", "--dropout", "0"], 1500),
    ("D_no_wd", ["--part", "D", "--weight-decay", "0"], 1500),
    ("D_no_aug", ["--part", "D", "--augment", "none"], 1500),
    ("E_lr1e-4", ["--part", "E", "--lr", "1e-4"], 1500),
    ("E_lr1e-2", ["--part", "E", "--lr", "1e-2"], 1500),
    ("E_warm1e-4", ["--part", "E", "--lr", "1e-4", "--warmup-epochs", "5"], 1500),
    ("E_warm1e-2", ["--part", "E", "--lr", "1e-2", "--warmup-epochs", "5"], 1500),
    ("F_focal", ["--part", "F", "--loss", "focal", "--focal-gamma", "2"], 1500),
]

PART_OF = {
    "ref": "0", "A_cnn": "A", "A_gru": "A", "B_conv0": "B", "B_conv1": "B",
    "B_conv2": "B", "C_w20": "C", "C_w42": "C", "C_w60": "C",
    "D_no_dropout": "D", "D_no_wd": "D", "D_no_aug": "D", "E_lr1e-4": "E",
    "E_lr1e-2": "E", "E_warm1e-4": "E", "E_warm1e-2": "E", "F_focal": "F",
}


def gpu_free_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        total, used = [int(v) for v in out.split(",")]
        return total - used
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--parts", nargs="*", default=None, help="restrict to these parts")
    parser.add_argument("--jobs", type=int, default=2, help="maximum concurrent runs")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--margin", type=int, default=250, help="free MiB kept in reserve")
    parser.add_argument("--list", action="store_true", help="print the job list and exit")
    args = parser.parse_args()

    jobs = []
    for seed in args.seeds:
        for name, extra, mem in VARIANTS:
            part = PART_OF[name]
            if args.parts and part not in args.parts:
                continue
            jobs.append((f"{name}_s{seed}", extra + ["--seed", str(seed)], mem, seed, name))

    if args.list:
        for run_id, extra, mem, seed, name in jobs:
            print(f"{run_id:<18s} mem~{mem:>4d} MiB  {' '.join(extra)}")
        return

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    running: Dict[int, Tuple[str, int]] = {}

    def reap() -> None:
        for pid in list(running):
            if not _alive(pid):
                run_id = running.pop(pid)[0]
                print(f"[done] {run_id}", flush=True)

    for run_id, extra, mem, seed, name in jobs:
        run_dir = ROOT / "runs" / run_id
        if (run_dir / "run.json").exists():
            print(f"[skip] {run_id} (finished)", flush=True)
            continue

        while True:
            reap()
            free = gpu_free_mb()
            if len(running) < args.jobs and free >= mem + args.margin:
                break
            time.sleep(15)

        cmd = [
            PY, "train.py", "--run-id", run_id,
            "--epochs", str(args.epochs), "--patience", str(args.patience),
        ]
        if (run_dir / "best.pt").exists():
            cmd.append("--resume")
        cmd += extra
        log_file = log_dir / f"{run_id}.log"
        print(f"[start] {run_id} (free {gpu_free_mb()} MiB) -> {log_file}", flush=True)
        with log_file.open("w") as fh:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        running[proc.pid] = (run_id, mem)

    while running:
        reap()
        time.sleep(15)
    print("[all done]", flush=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
