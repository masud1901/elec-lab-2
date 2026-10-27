"""Generate notebooks/run_ablation_colab.ipynb (no manual JSON editing)."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


md(
    """# DeepConvLSTM ablation study — Google Colab runner

Runs part of the ablation matrix on a Colab GPU (T4 / L4 / A100), then brings
the results back as a zip (and optionally pushes them to GitHub).

Useful when the local machine is busy: each Colab session can take a different
part of the matrix, and the per-run `run.json` / `per_class.csv` files are
merged by `summarize.py` afterwards.
"""
)

code(
    """# --- configuration -------------------------------------------------------
REPO_URL = "https://github.com/masud1901/elec-lab-2.git"   # private repo: set GITHUB_TOKEN below
BRANCH = "master"
SEEDS = [42]            # e.g. [42, 43, 44]
PARTS = ["A", "B"]       # 0 (reference), A, B, C, D, E, F
JOBS = 1                # concurrent trainings on one GPU (2 needs >8 GB)
EPOCHS = 80
PUSH_BACK = False       # push results to GitHub with GITHUB_TOKEN
GITHUB_TOKEN = ""       # paste a token with repo scope if you want to push
"""
)

code(
    """!pip -q install pandas scikit-learn matplotlib pyyaml tqdm pytest weasyprint markdown

import subprocess, os, sys, json, glob
from pathlib import Path

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, "elec-lab-2"],
               check=False, cwd="/content")
os.chdir("/content/elec-lab-2")

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""
)

code(
    """# --- data ---------------------------------------------------------------
!python download_data.py
!python preprocess.py --split paper
"""
)

code(
    """# --- run the matrix ------------------------------------------------------
cmd = ["python", "scripts/run_matrix.py", "--seeds", *[str(s) for s in SEEDS],
       "--jobs", str(JOBS), "--epochs", str(EPOCHS)]
if PARTS:
    cmd += ["--parts", *PARTS]
print(" ".join(cmd))
subprocess.run(cmd, check=False)
"""
)

code(
    """# --- summarise and build the report --------------------------------------
subprocess.run(["python", "summarize.py"], check=False)
subprocess.run(["python", "report/fill_report.py"], check=False)
subprocess.run(["python", "report/build_report.py"], check=False)

import pandas as pd
print(pd.read_csv("results/agg.csv")[["part", "variant", "n_seeds",
      "test_weighted_f1_mean", "test_macro_f1_mean"]].to_string(index=False))
"""
)

code(
    """# --- collect the results -------------------------------------------------
import shutil, zipfile, os

shutil.make_archive("/content/ablation_results", "zip", "/content/elec-lab-2/results")
for run_dir in glob.glob("runs/*/"):
    os.makedirs(f"/content/collected/{run_dir}", exist_ok=True)
    for name in ("run.json", "history.csv", "per_class.csv", "confusion_test.npy", "config.json"):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy(src, f"/content/collected/{run_dir}/{name}")
shutil.make_archive("/content/ablation_runs", "zip", "/content/collected")
print(os.path.getsize("/content/ablation_results.zip") / 1e6, "MB of tables and figures")
"""
)

code(
    """# --- optional: push the results back to GitHub ---------------------------
if PUSH_BACK and GITHUB_TOKEN:
    subprocess.run(["git", "config", "user.email", "masud1901@users.noreply.github.com"])
    subprocess.run(["git", "config", "user.name", "Masud"])
    subprocess.run(["git", "add", "results"], check=False)
    subprocess.run(["git", "commit", "-m", "Colab results"], check=False)
    url = REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")
    subprocess.run(["git", "push", url, BRANCH], check=False)
    print("pushed")
else:
    print("skipping push (PUSH_BACK=False or no token)")
"""
)

code(
    """# --- download ------------------------------------------------------------
from google.colab import files
files.download("/content/ablation_results.zip")
files.download("/content/ablation_runs.zip")
"""
)

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "name": "run_ablation_colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "run_ablation_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
