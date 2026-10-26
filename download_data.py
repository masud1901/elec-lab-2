"""Download and unpack the OPPORTUNITY activity recognition dataset.

The UCI archive is ~300 MB compressed.  The script is idempotent: if the
archive and the extracted files are already present it does nothing.

Usage:
    python download_data.py                # data/raw
    python download_data.py --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URLS = [
    "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip",
]

EXPECTED_RUNS = [
    f"{s}-{r}"
    for s in ("S1", "S2", "S3", "S4")
    for r in ("ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill")
]


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        start = time.time()
        with tmp.open("wb") as f:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                f.write(block)
                downloaded += len(block)
                if total:
                    pct = 100.0 * downloaded / total
                    speed = downloaded / max(time.time() - start, 1e-6) / 1e6
                    print(
                        f"\r  {downloaded/1e6:7.1f}/{total/1e6:.1f} MB "
                        f"({pct:5.1f}%) {speed:5.1f} MB/s",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  {downloaded/1e6:7.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def extract(zip_path: Path, out_dir: Path, force: bool = False) -> None:
    dataset_dir = out_dir / "OpportunityUCIDataset" / "dataset"
    if dataset_dir.exists() and not force:
        missing = [r for r in EXPECTED_RUNS if not (dataset_dir / f"{r}.dat").exists()]
        if not missing:
            print(f"[extract] already extracted: {dataset_dir}")
            return
        print(f"[extract] missing runs: {missing}; re-extracting")

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        needed = [
            n
            for n in names
            if n.startswith("OpportunityUCIDataset/dataset/")
            and (n.endswith(".dat") or n.endswith(".txt"))
        ]
        print(f"[extract] {len(needed)} files -> {out_dir}")
        for name in needed:
            zf.extract(name, out_dir)


def verify(data_dir: Path) -> None:
    dataset_dir = data_dir / "OpportunityUCIDataset" / "dataset"
    missing = [r for r in EXPECTED_RUNS if not (dataset_dir / f"{r}.dat").exists()]
    if missing:
        raise SystemExit(f"Missing dataset files: {missing}")
    for name in ("column_names.txt", "label_legend.txt"):
        if not (dataset_dir / name).exists():
            raise SystemExit(f"Missing {name}")
    print(f"[verify] found {len(EXPECTED_RUNS)} recordings in {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="directory for the raw archive")
    parser.add_argument("--url", default=None, help="override download URL")
    parser.add_argument("--force", action="store_true", help="re-download / re-extract")
    args = parser.parse_args()

    out_dir = Path(args.out)
    zip_path = out_dir / "OpportunityUCIDataset.zip"

    if args.force and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        urls = [args.url] if args.url else URLS
        errors = []
        for url in urls:
            try:
                download(url, zip_path)
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                errors.append(f"{url}: {exc}")
                if zip_path.exists():
                    zip_path.unlink()
        else:
            raise SystemExit("Could not download dataset:\n" + "\n".join(errors))
    else:
        print(f"[download] using cached {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")

    if not zipfile.is_zipfile(zip_path):
        raise SystemExit(f"Not a valid zip file: {zip_path}")

    extract(zip_path, out_dir, force=args.force)
    verify(out_dir)

    digest = sha256sum(zip_path)
    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        f"archive={zip_path.name}\n"
        f"size_bytes={zip_path.stat().st_size}\n"
        f"sha256={digest}\n"
        f"runs={','.join(EXPECTED_RUNS)}\n"
    )
    print(f"[manifest] sha256={digest}")
    print("[done] dataset ready")


if __name__ == "__main__":
    main()
