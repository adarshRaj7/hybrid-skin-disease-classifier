#!/usr/bin/env python3
"""Dataset setup -- a separate, explicit step from training.

The original notebook embedded a Kaggle username and API key directly in a
shell command. Those specific credentials must be treated as compromised
(rotate/revoke them on kaggle.com if you have not already) and this project
never stores credentials in code, config, or notebooks again.

This script instead reads Kaggle credentials from the standard environment
variables the `kaggle` CLI already supports:

    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    python scripts/prepare_dataset.py --output-dir data

Or place a kaggle.json file yourself at ~/.kaggle/kaggle.json (chmod 600) and
omit the environment variables -- either way, this script never writes or
reads a secret from anywhere inside the repository.

After downloading, it also runs an explicit, offline image-validation pass
and reports (but does not silently delete) unreadable files.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skin_disease.dataset import validate_directory  # noqa: E402

KAGGLE_DATASET_SLUG = "pacificrm/skindiseasedataset"


def download_dataset(output_dir: Path) -> Path:
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            raise SystemExit(
                "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "environment variables, or place ~/.kaggle/kaggle.json yourself. "
                "This script will not accept credentials as a command-line argument."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "skindiseasedataset.zip"

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_SLUG, "-p", str(output_dir)],
        check=True,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    zip_path.unlink(missing_ok=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate the skin disease dataset.")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--skip-download", action="store_true", help="Only run validation on an existing directory.")
    parser.add_argument("--train-subdir", type=str, default="SkinDisease/SkinDisease/train")
    parser.add_argument("--test-subdir", type=str, default="SkinDisease/SkinDisease/test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not args.skip_download:
        download_dataset(output_dir)

    for subdir in (args.train_subdir, args.test_subdir):
        full_path = output_dir / subdir
        if not full_path.exists():
            print(f"WARNING: expected directory not found: {full_path}")
            continue
        unreadable, total = validate_directory(full_path)
        print(f"{full_path}: {total - len(unreadable)}/{total} images readable")
        if unreadable:
            print(f"  {len(unreadable)} unreadable file(s), e.g.:")
            for p in unreadable[:10]:
                print(f"    {p}")
            print("  These were reported, not deleted. Remove them manually if appropriate.")


if __name__ == "__main__":
    main()
