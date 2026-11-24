"""Dataset loading, label mapping and split construction on synthetic data."""
import pandas as pd
import pytest
import torch

from src.data.ben_s2_dataset import BigEarthNetS2RGBDataset, build_label_to_index
from src.data import create_ben_s2_first_split as split_mod

from conftest import LABELS


def test_label_mapping_is_sorted_and_complete(synthetic_splits):
    split_dir, _ = synthetic_splits
    mapping = build_label_to_index(split_dir)
    assert list(mapping) == sorted(LABELS) and list(mapping.values()) == list(range(len(LABELS)))


def test_dataset_item_shapes_scaling_and_labels(synthetic_splits):
    split_dir, root = synthetic_splits
    mapping = build_label_to_index(split_dir)
    ds = BigEarthNetS2RGBDataset(split_dir / "train_seen.csv", mapping, image_size=64, data_root=root)
    image, label, meta = ds[2]
    assert image.shape == (3, 64, 64) and image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert label.tolist() == [1.0, 1.0, 1.0] and meta["patch_id"] == "train_seen_2"


def test_missing_band_file_fails_clearly(synthetic_splits):
    split_dir, root = synthetic_splits
    mapping = build_label_to_index(split_dir)
    ds = BigEarthNetS2RGBDataset(split_dir / "train_seen.csv", mapping, data_root=root / "nowhere")
    with pytest.raises(FileNotFoundError):
        ds[0]


def _synthetic_metadata(path):
    rows = []
    cells = split_mod.TRAIN_CELLS + [(split_mod.HELDOUT_TILE, split_mod.HELDOUT_SEASON), ("T99XXX", "winter")]
    for c, (tile, season) in enumerate(cells):
        for i in range(20):
            pid = f"S2A_MSIL2A_20180101T000000_N9999_R001_{tile}_{c}_{i}"
            rows.append({"patch_id": pid, "patch_dir": f"p/{pid}", "B02_path": "x", "B03_path": "x", "B04_path": "x",
                         "labels": "Pastures|Urban fabric" if i % 2 else "Pastures", "tile_id": tile,
                         "date": "20180101", "month": 1, "season": season})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_first_split_is_disjoint_and_holds_out_one_cell(tmp_path):
    meta = tmp_path / "meta.csv"
    _synthetic_metadata(meta)
    split_mod.create_split(meta, tmp_path / "splits", seed=0)
    s = {n: pd.read_csv(tmp_path / "splits" / f"{n}.csv") for n in
         ["train_seen", "val_seen", "test_unseen", "oracle_train_unseen", "oracle_test_unseen"]}
    ids = {n: set(df["patch_id"]) for n, df in s.items()}
    held = (split_mod.HELDOUT_TILE, split_mod.HELDOUT_SEASON)

    assert ids["train_seen"].isdisjoint(ids["val_seen"])
    assert ids["train_seen"].isdisjoint(ids["test_unseen"]) and ids["val_seen"].isdisjoint(ids["test_unseen"])
    assert set(zip(s["test_unseen"].tile_id, s["test_unseen"].season)) == {held}
    assert held not in set(zip(s["train_seen"].tile_id, s["train_seen"].season))
    assert ("T99XXX", "winter") not in set(zip(pd.concat(s.values()).tile_id, pd.concat(s.values()).season))
    # oracle halves partition the held-out cell
    assert ids["oracle_train_unseen"].isdisjoint(ids["oracle_test_unseen"])
    assert ids["oracle_train_unseen"] | ids["oracle_test_unseen"] == ids["test_unseen"]
    # the held-out tile itself IS seen in training (other seasons): compositional, not spatial, hold-out
    assert split_mod.HELDOUT_TILE in set(s["train_seen"].tile_id)
