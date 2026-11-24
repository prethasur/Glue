"""Shared synthetic fixtures. No real BigEarthNet data or network access is used."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SPLITS = ["train_seen", "val_seen", "test_unseen", "oracle_train_unseen", "oracle_test_unseen"]
LABELS = ["Coniferous forest", "Pastures", "Urban fabric"]


def write_synthetic_splits(root: Path, n_per_split: int = 4, size: int = 32) -> Path:
    """Tiny 3-band uint16 TIFF patches and split CSVs with relative band paths."""
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(0)
    split_dir = root / "splits"
    split_dir.mkdir(parents=True)
    for s_idx, split in enumerate(SPLITS):
        rows = []
        for i in range(n_per_split):
            pid = f"{split}_{i}"
            pdir = root / "patches" / pid
            pdir.mkdir(parents=True)
            for band in ("B04", "B03", "B02"):
                tifffile.imwrite(pdir / f"{pid}_{band}.tif", rng.integers(0, 10000, (size, size), dtype=np.uint16))
            rows.append({"patch_id": pid, "tile_id": f"T{s_idx}", "season": "spring",
                         "labels": "|".join(LABELS[: 1 + i % 3]),
                         "B04_path": f"patches/{pid}/{pid}_B04.tif",
                         "B03_path": f"patches/{pid}/{pid}_B03.tif",
                         "B02_path": f"patches/{pid}/{pid}_B02.tif"})
        with open(split_dir / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return split_dir


@pytest.fixture
def synthetic_splits(tmp_path):
    return write_synthetic_splits(tmp_path), tmp_path
