"""PyTorch dataset for BigEarthNet-S2 RGB split CSVs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


SPLIT_NAMES = [
    "train_seen",
    "val_seen",
    "test_unseen",
    "oracle_train_unseen",
    "oracle_test_unseen",
]
LABEL_SEPARATOR = "|"


def parse_labels(labels: str) -> list[str]:
    return [label.strip() for label in str(labels).split(LABEL_SEPARATOR) if label.strip()]


def build_label_to_index(split_dir: str | Path, splits: list[str] | None = None) -> dict[str, int]:
    split_dir = Path(split_dir)
    splits = splits or ["train_seen", "val_seen", "test_unseen"]
    labels: set[str] = set()
    for split in splits:
        csv_path = split_dir / f"{split}.csv"
        df = pd.read_csv(csv_path, usecols=["labels"])
        for label_string in df["labels"].fillna(""):
            labels.update(parse_labels(label_string))
    return {label: index for index, label in enumerate(sorted(labels))}


def save_label_to_index(label_to_index: dict[str, int], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(label_to_index, indent=2, sort_keys=True), encoding="utf-8")


def load_label_to_index(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_data_root(data_root: str | Path | None = None) -> Path | None:
    root = data_root or os.environ.get("BEN_S2_DATA_ROOT")
    if root is None or str(root).strip() == "":
        return None
    return Path(root).expanduser().resolve()


def resolve_band_path(row: pd.Series, band: str, data_root: Path | None = None) -> str:
    path_column = f"{band}_path"
    original = Path(str(row[path_column]))
    if original.is_file():
        return str(original)

    if data_root is None:
        raise FileNotFoundError(
            f"{path_column} does not exist and no data_root/BEN_S2_DATA_ROOT was provided: {original}"
        )

    candidates: list[Path] = []
    if not original.is_absolute():
        candidates.append(data_root / original)

    patch_id = str(row["patch_id"])
    patch_dir_value = str(row.get("patch_dir", ""))
    if patch_dir_value:
        patch_dir = Path(patch_dir_value)
        if patch_dir.is_absolute():
            candidates.append(data_root / patch_dir.name)
            candidates.append(data_root / "patches" / patch_dir.name)
        else:
            candidates.append(data_root / patch_dir)
            candidates.append(data_root / "patches" / patch_dir.name)

    candidates.append(data_root / patch_id / f"{patch_id}_{band}.tif")
    candidates.append(data_root / "patches" / patch_id / f"{patch_id}_{band}.tif")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not resolve {path_column} for patch_id={patch_id}. "
        f"Original={original}; data_root={data_root}"
    )


def read_rgb_tensor(b04_path: str, b03_path: str, b02_path: str, image_size: int = 224) -> torch.Tensor:
    channels = []
    for path in (b04_path, b03_path, b02_path):
        band = tifffile.imread(path).astype(np.float32)
        channels.append(band)
    image = np.stack(channels, axis=0)

    # Sentinel-2 L2A reflectance is commonly scaled by 10000. Clamp keeps clouds/outliers tame.
    image = np.clip(image, 0.0, 10000.0) / 10000.0
    tensor = torch.from_numpy(image)
    tensor = F.resize(
        tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return tensor.to(torch.float32)


class BigEarthNetS2RGBDataset(Dataset):
    """Loads BigEarthNet-S2 split CSV rows as RGB tensors and multi-hot labels."""

    def __init__(
        self,
        csv_path: str | Path,
        label_to_index: dict[str, int],
        image_size: int = 224,
        data_root: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rows = pd.read_csv(self.csv_path)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.data_root = resolve_data_root(data_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows.iloc[index]
        image = read_rgb_tensor(
            b04_path=resolve_band_path(row, "B04", self.data_root),
            b03_path=resolve_band_path(row, "B03", self.data_root),
            b02_path=resolve_band_path(row, "B02", self.data_root),
            image_size=self.image_size,
        )
        label = torch.zeros(len(self.label_to_index), dtype=torch.float32)
        for label_name in parse_labels(row["labels"]):
            if label_name in self.label_to_index:
                label[self.label_to_index[label_name]] = 1.0

        metadata = {
            "patch_id": str(row["patch_id"]),
            "tile_id": str(row["tile_id"]),
            "season": str(row["season"]),
            "cell_id": f"{row['tile_id']}_{row['season']}",
        }
        return image, label, metadata
