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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = os.environ.get("PY") or (str(_VENV_PY) if _VENV_PY.exists() else sys.executable)

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

# Variants the workshop PDF does not ask for (B: only 1 and 2 conv layers,
# C: only 0.67 s and 2 s windows).
NOT_IN_WORKSHOP = {"B_conv0", "C_w42"}

PART_OF = {
    "ref": "0", "A_cnn": "A", "A_gru": "A", "B_conv0": "B", "B_conv1": "B",
    "B_conv2": "B", "C_w20": "C", "C_w42": "C", "C_w60": "C",
    "D_no_dropout": "D", "D_no_wd": "D", "D_no_aug": "D", "E_lr1e-4": "E",
    "E_lr1e-2": "E", "E_warm1e-4": "E", "E_warm1e-2": "E", "F_focal": "F",
}


def gpu_free_mb() -> Optional[int]:
    """Free VRAM in MiB, or None if it cannot be determined."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        ).strip().splitlines()[0]
        total, used = [int(v.strip()) for v in out.split(",")[:2]]
        return max(total - used, 0)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] cannot read GPU memory: {exc}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--parts", nargs="*", default=None, help="restrict to these parts")
    parser.add_argument("--exclude-parts", nargs="*", default=None,
                        help="skip these parts (e.g. when another machine runs them)")
    parser.add_argument("--jobs", type=int, default=2, help="maximum concurrent runs")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--margin", type=int, default=600,
                        help="free MiB kept in reserve (each run also holds its data on the GPU)")
    parser.add_argument("--workshop", action="store_true",
                        help="only the runs the workshop PDF asks for (skips B_conv0, C_w42)")
    parser.add_argument("--list", action="store_true", help="print the job list and exit")
    parser.add_argument("--quiet", action="store_true", help="do not echo training output")
    parser.add_argument("--order", default="heavy-first",
                        choices=["heavy-first", "listed"],
                        help="start the most expensive runs first (they are the long poles)")
    parser.add_argument("--status-every", type=int, default=120,
                        help="seconds between GPU status lines")
    parser.add_argument("--echo-every", type=int, default=1,
                        help="print every n-th epoch line of each run (all lines go to logs/)")
    parser.add_argument("--ramp-up", type=int, default=45,
                        help="seconds during which a started job's VRAM counts as reserved")
    parser.add_argument("--max-wait", type=int, default=900,
                        help="max seconds to wait for free VRAM before starting anyway")
    args = parser.parse_args()

    jobs = []
    for seed in args.seeds:
        for name, extra, mem in VARIANTS:
            part = PART_OF[name]
            if args.parts and part not in args.parts:
                continue
            if args.exclude_parts and part in args.exclude_parts:
                continue
            if args.workshop and name in NOT_IN_WORKSHOP:
                continue
            jobs.append((f"{name}_s{seed}", extra + ["--seed", str(seed)], mem, seed, name))

    if args.order == "heavy-first":
        # longest runs first so they are not left over at the end of a session
        jobs.sort(key=lambda j: -j[2])

    if args.list:
        for run_id, extra, mem, seed, name in jobs:
            print(f"{run_id:<18s} mem~{mem:>4d} MiB  {' '.join(extra)}")
        return

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    running: List[Job] = []

    print(f"[runner] python  = {PY} (exists={Path(PY).exists()})", flush=True)
    print(f"[runner] repo    = {ROOT}", flush=True)
    print(f"[runner] free VRAM = {gpu_free_mb()} MiB, cpus = {os.cpu_count()}", flush=True)
    print(f"[runner] {len(jobs)} runs, up to {args.jobs} at a time", flush=True)
    if (os.cpu_count() or 1) < args.jobs:
        print(f"[warn] --jobs {args.jobs} > {os.cpu_count()} CPUs: the trainings will compete "
              "for CPU time; --jobs 2 is usually as fast", flush=True)
    processed = ROOT / "data" / "processed"
    if not processed.exists():
        print(
            "[warn] data/processed is missing -> run: python preprocess.py --split paper",
            flush=True,
        )
    if not (ROOT / "train.py").exists():
        raise SystemExit(f"train.py not found in {ROOT} (wrong working directory?)")

    failed: List[str] = []

    def reap() -> None:
        # Popen.poll() also collects the exit status; os.kill(pid, 0) would keep
        # reporting finished children (zombies) as alive and block the queue.
        for job in list(running):
            code = job.proc.poll()
            if code is None:
                continue
            running.remove(job)
            job.thread.join(timeout=5)
            took = human_time(time.time() - job.started)
            if code == 0:
                print(f"[done] {job.run_id} in {took}", flush=True)
            else:
                failed.append(job.run_id)
                print(f"[FAILED] {job.run_id} exit code {code} after {took}; "
                      f"see logs/{job.run_id}.log", flush=True)

    def reserved_mb() -> int:
        """VRAM that recently started jobs have not allocated yet."""
        now = time.time()
        return sum(j.mem for j in running if now - j.started < args.ramp_up)

    def stop_all() -> None:
        for job in running:
            if job.proc.poll() is None:
                job.proc.send_signal(signal.SIGINT)
        deadline = time.time() + 20
        for job in running:
            try:
                job.proc.wait(timeout=max(deadline - time.time(), 0.1))
            except subprocess.TimeoutExpired:
                job.proc.kill()

    try:
        for run_id, extra, mem, seed, name in jobs:
            run_dir = ROOT / "runs" / run_id
            if (run_dir / "run.json").exists():
                print(f"[skip] {run_id} (finished)", flush=True)
                continue

            last_status = 0.0
            wait_since = time.time()
            while True:
                reap()
                free = gpu_free_mb()
                if free is not None:
                    free -= reserved_mb()
                memory_ok = free is None or free >= mem + args.margin
                wait_too_long = time.time() - wait_since > args.max_wait
                if len(running) < args.jobs and (memory_ok or wait_too_long):
                    if wait_too_long and not memory_ok:
                        print(
                            f"[warn] waited {args.max_wait}s for VRAM (need "
                            f"{mem + args.margin} MiB, free {free}); starting anyway",
                            flush=True,
                        )
                    break
                now = time.time()
                if now - last_status > args.status_every:
                    last_status = now
                    active = ", ".join(f"{j.run_id}({j.mem} MiB)" for j in running)
                    print(
                        f"[wait] free {free} MiB, need {mem + args.margin} MiB | "
                        f"running: {active or '-'}",
                        flush=True,
                    )
                time.sleep(2)

            cmd = [
                PY, "-u", "train.py", "--run-id", run_id,
                "--epochs", str(args.epochs), "--patience", str(args.patience),
            ]
            if (run_dir / "best.pt").exists():
                cmd.append("--resume")
            cmd += extra
            log_file = log_dir / f"{run_id}.log"
            print(f"[start] {run_id} (free {gpu_free_mb()} MiB) -> {log_file}", flush=True)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            # the pump thread must always drain the pipe, otherwise the child
            # blocks once the pipe buffer is full (also with --quiet)
            thread = threading.Thread(
                target=_stream,
                args=(proc, log_file, run_id, not args.quiet, args.echo_every),
                daemon=True,
            )
            thread.start()
            running.append(Job(run_id, mem, proc, thread, time.time()))

        while running:
            reap()
            time.sleep(2)
    except KeyboardInterrupt:
        print("[runner] interrupted: stopping running trainings", flush=True)
        stop_all()
        raise SystemExit(130)

    if failed:
        print(f"[all done] {len(failed)} run(s) failed: {', '.join(failed)}", flush=True)
        raise SystemExit(1)
    print("[all done]", flush=True)


class Job:
    def __init__(self, run_id, mem, proc, thread, started):
        self.run_id = run_id
        self.mem = mem
        self.proc = proc
        self.thread = thread
        self.started = started


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _stream(proc, log_path: Path, run_id: str, echo: bool, echo_every: int) -> None:
    """Write the training output of one run to its log file and (optionally) stdout.

    ``echo_every`` > 1 prints only every n-th epoch line, which keeps the
    notebook output readable when several runs train at once; the log file
    always has every line.
    """
    n_epoch_lines = 0
    with log_path.open("w") as fh:
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if not echo:
                continue
            if "[epoch" in line:
                n_epoch_lines += 1
                if n_epoch_lines % max(echo_every, 1) != 1 and echo_every > 1:
                    continue
            if len(line) > 200:
                line = line[:197] + "...\n"
            sys.stdout.write(f"[{run_id}] {line}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
