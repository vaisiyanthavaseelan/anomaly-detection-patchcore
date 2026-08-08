"""Extracts metal_nut/ and screw/ from the official MVTec-AD archive into data/.

You must download the archive yourself from
https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads/
(requires filling out MVTec's short form and accepting their research-use
license -- this script only handles local extraction).

Usage:
    ./venv/bin/python scripts/prepare_data.py --archive ~/Downloads/mvtec_anomaly_detection.tar.xz
"""
import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CATEGORIES, DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path, help="Path to mvtec_anomaly_detection.tar.xz")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = parser.parse_args()

    if not args.archive.exists():
        raise FileNotFoundError(f"Archiv nicht gefunden: {args.archive}")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.archive, "r:xz") as tar:
        members = [
            m for m in tar.getmembers()
            if any(m.name == c or m.name.startswith(f"{c}/") for c in args.categories)
        ]
        print(f"Entpacke {len(members)} Dateien fuer Kategorien {args.categories} ...")
        tar.extractall(path=DATA_ROOT, members=members)

    for category in args.categories:
        extracted = DATA_ROOT / category
        if extracted.exists():
            print(f"  {category}: OK ({extracted})")
        else:
            print(f"  {category}: WARNUNG - nicht im Archiv gefunden")


if __name__ == "__main__":
    main()
