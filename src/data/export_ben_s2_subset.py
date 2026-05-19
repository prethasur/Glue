"""Export a portable BigEarthNet-S2 subset for the first split."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.data.ben_s2_dataset import SPLIT_NAMES


def portable_patch_dir(patch_id: str) -> str:
    return f"patches/{patch_id}"


def portable_band_path(patch_id: str, band: str) -> str:
    return f"patches/{patch_id}/{patch_id}_{band}.tif"


def copy_patch_folder(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(source, destination)


def export_subset(split_dir: Path, subset_root: Path) -> None:
    patches_root = subset_root / "patches"
    portable_split_dir = subset_root / "splits"
    patches_root.mkdir(parents=True, exist_ok=True)
    portable_split_dir.mkdir(parents=True, exist_ok=True)

    patch_sources: dict[str, Path] = {}
    split_counts: dict[str, int] = {}
    rewritten: dict[str, pd.DataFrame] = {}

    for split_name in SPLIT_NAMES:
        csv_path = split_dir / f"{split_name}.csv"
        df = pd.read_csv(csv_path, dtype={"date": str})
        split_counts[split_name] = len(df)
        for row in df.itertuples(index=False):
            patch_id = str(row.patch_id)
            source = Path(str(row.patch_dir))
            if not source.is_dir():
                source = Path(str(row.B02_path)).parent
            if not source.is_dir():
                raise FileNotFoundError(f"Could not find source patch folder for {patch_id}: {source}")
            patch_sources.setdefault(patch_id, source)

        df = df.copy()
        df["patch_dir"] = df["patch_id"].map(portable_patch_dir)
        df["B02_path"] = df["patch_id"].map(lambda patch_id: portable_band_path(str(patch_id), "B02"))
        df["B03_path"] = df["patch_id"].map(lambda patch_id: portable_band_path(str(patch_id), "B03"))
        df["B04_path"] = df["patch_id"].map(lambda patch_id: portable_band_path(str(patch_id), "B04"))
        rewritten[split_name] = df

    for patch_id, source in sorted(patch_sources.items()):
        copy_patch_folder(source, patches_root / patch_id)

    for split_name, df in rewritten.items():
        df.to_csv(portable_split_dir / f"{split_name}.csv", index=False)

    label_json = split_dir / "label_to_index.json"
    if label_json.is_file():
        shutil.copy2(label_json, portable_split_dir / "label_to_index.json")

    print(f"unique patches copied: {len(patch_sources)}")
    print("split sample counts:")
    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count}")
    print(f"portable patches: {patches_root}")
    print(f"portable splits: {portable_split_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_dir",
        default=Path("outputs/ben_s2_splits_first"),
        type=Path,
        help="Directory containing the original split CSVs.",
    )
    parser.add_argument(
        "--subset_root",
        default=Path("outputs/ben_s2_first_subset"),
        type=Path,
        help="Output root for portable patches and rewritten split CSVs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir.expanduser().resolve()
    subset_root = args.subset_root.expanduser().resolve()
    if not split_dir.is_dir():
        print(f"ERROR: split_dir not found: {split_dir}")
        return 2
    export_subset(split_dir, subset_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
