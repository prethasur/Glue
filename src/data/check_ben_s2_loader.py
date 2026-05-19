"""Sanity-check the BigEarthNet-S2 RGB dataset loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data.ben_s2_dataset import (
    SPLIT_NAMES,
    BigEarthNetS2RGBDataset,
    build_label_to_index,
    save_label_to_index,
)


def print_batch_stats(split_name: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    label_freq = labels.sum(dim=0)
    nonzero = [(index, int(count.item())) for index, count in enumerate(label_freq) if count.item() > 0]
    print(f"{split_name}")
    print(f"  image tensor shape: {tuple(images.shape[1:])}")
    print(f"  label tensor shape: {tuple(labels.shape[1:])}")
    print(
        "  image batch min/max/mean: "
        f"{images.min().item():.6f} / {images.max().item():.6f} / {images.mean().item():.6f}"
    )
    print(f"  label frequency in one batch: {nonzero}")


def check_loader(split_dir: Path, batch_size: int, image_size: int, data_root: Path | None) -> None:
    label_to_index = build_label_to_index(split_dir)
    label_path = split_dir / "label_to_index.json"
    save_label_to_index(label_to_index, label_path)
    print(f"saved label mapping: {label_path}")
    print(f"label count: {len(label_to_index)}")

    montage_saved = False
    for split_name in SPLIT_NAMES:
        csv_path = split_dir / f"{split_name}.csv"
        dataset = BigEarthNetS2RGBDataset(
            csv_path,
            label_to_index,
            image_size=image_size,
            data_root=data_root,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        images, labels, _metadata = next(iter(loader))
        print(f"\n{split_name} samples: {len(dataset)}")
        print_batch_stats(split_name, images, labels)

        if not montage_saved:
            montage_path = split_dir / "rgb_loader_montage.png"
            save_image(images, montage_path, nrow=min(batch_size, 4))
            print(f"  montage saved: {montage_path}")
            montage_saved = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_dir",
        default=Path("outputs/ben_s2_splits_first"),
        type=Path,
        help="Directory containing BigEarthNet-S2 split CSVs.",
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument(
        "--data_root",
        default=None,
        type=Path,
        help="Optional BigEarthNet-S2 root or exported subset root. Can also use BEN_S2_DATA_ROOT.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir.expanduser().resolve()
    if not split_dir.is_dir():
        print(f"ERROR: split directory not found: {split_dir}")
        return 2
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    check_loader(split_dir, args.batch_size, args.image_size, data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
