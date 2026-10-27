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
REPO_URL = "https://github.com/masud1901/elec-lab-2.git"
BRANCH = "master"
SEEDS = [42]            # e.g. [42, 43, 44]
PARTS = ["A", "B"]       # 0 (reference), A, B, C, D, E, F
JOBS = 1                # concurrent trainings on one GPU (2 needs >8 GB)
EPOCHS = 80
PUSH_BACK = False       # push results to GitHub (needs a token, repo is public)

# --- GitHub access ---------------------------------------------------------
# The repository is PUBLIC, so cloning works without any token.
# A token is only needed if PUSH_BACK = True (write access).
# Create one at github.com/settings/tokens with scope "repo", paste it here,
# or store it as a Colab secret named GITHUB_TOKEN.
GITHUB_TOKEN = ""

try:                                   # Colab secrets
    from google.colab import userdata
    GITHUB_TOKEN = GITHUB_TOKEN or userdata.get("GITHUB_TOKEN") or ""
except Exception:
    try:                               # local Jupyter/VS Code
        import subprocess
        GITHUB_TOKEN = GITHUB_TOKEN or subprocess.check_output(
            ["gh", "auth", "token"], text=True).strip()
    except Exception:
        pass

print("token found:", bool(GITHUB_TOKEN), "(not needed for cloning a public repo)")
"""
)

code(
    """!pip -q install pandas scikit-learn matplotlib pyyaml tqdm pytest weasyprint markdown

import os, sys, json, glob, shutil, subprocess, zipfile
from pathlib import Path

HERE = Path.cwd()
IN_REPO = (HERE / "train.py").exists() and (HERE / "scripts").is_dir()

if IN_REPO:
    # already inside the checkout (e.g. VS Code / local Jupyter): use it as is
    REPO_DIR = HERE
    print("using existing checkout:", REPO_DIR)
elif Path("/content").exists():
    REPO_DIR = Path("/content/elec-lab-2")
else:
    REPO_DIR = Path.home() / "elec-lab-2-run"

def clone_repo():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    urls = []
    if GITHUB_TOKEN:
        urls.append(REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@"))
    urls.append(REPO_URL)
    for url in urls:
        r = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", BRANCH, url, str(REPO_DIR)],
            capture_output=True, text=True)
        if REPO_DIR.exists():
            print("cloned", REPO_DIR)
            return True
        err = r.stderr.strip().splitlines()
        print("clone failed:", err[-1] if err else r.returncode)
    return False

if not IN_REPO and not clone_repo():
    raise SystemExit(
        "Could not clone the repository.\\n"
        "Fix one of these:\\n"
        "  1) make the GitHub repo public, or\\n"
        "  2) set GITHUB_TOKEN in the configuration cell (scope: repo), or\\n"
        "  3) upload elec-lab-2.zip with the file uploader and unzip it to /content.")

os.chdir(REPO_DIR)
print("working dir:", os.getcwd())

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
    """# --- check the GPU ------------------------------------------------------
!nvidia-smi
"""
)

code(
    """# --- run the matrix ------------------------------------------------------
# Output of every training run is echoed live (one line per epoch) and also
# written to logs/<run-id>.log
cmd = ["python", "-u", "scripts/run_matrix.py", "--seeds", *[str(s) for s in SEEDS],
       "--jobs", str(JOBS), "--epochs", str(EPOCHS)]
if PARTS:
    cmd += ["--parts", *PARTS]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=False)
"""
)

code(
    """# --- watch progress (optional, run in a second cell) ---------------------
# Streams the live log of the running trainings; Ctrl-C stops the tail only.
!tail -n 30 -f logs/*.log
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
shutil.make_archive("/content/ablation_results" if Path("/content").exists() else "ablation_results",
                    "zip", str(REPO_DIR / "results"))
collected = Path("/content/collected") if Path("/content").exists() else Path("collected")
shutil.rmtree(collected, ignore_errors=True)
for run_dir in glob.glob(str(REPO_DIR / "runs/*/")):
    target = collected / Path(run_dir).name
    target.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "history.csv", "per_class.csv", "confusion_test.npy", "config.json"):
        src = Path(run_dir) / name
        if src.exists():
            shutil.copy(src, target / name)
base = "/content/ablation_runs" if Path("/content").exists() else "ablation_runs"
shutil.make_archive(base, "zip", str(collected))
print(base + ".zip created")
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
