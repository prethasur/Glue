"""CPU smoke test on a tiny synthetic BigEarthNet-S2-like dataset (no real data needed).

Run from the repository root:  python -m pytest tests
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
REPO = Path(__file__).resolve().parents[1]
SPLITS = ["train_seen", "val_seen", "test_unseen", "oracle_train_unseen", "oracle_test_unseen"]
LABELS = ["Coniferous forest", "Pastures", "Urban fabric"]


def _make_dataset(root: Path) -> Path:
    rng = np.random.default_rng(0)
    split_dir = root / "splits"
    split_dir.mkdir()
    for s_idx, split in enumerate(SPLITS):
        rows = []
        for i in range(4):
            pid = f"{split}_{i}"
            pdir = root / "patches" / pid
            pdir.mkdir(parents=True)
            for band in ("B04", "B03", "B02"):
                tifffile.imwrite(pdir / f"{pid}_{band}.tif",
                                 rng.integers(0, 10000, (32, 32), dtype=np.uint16))
            rows.append({"patch_id": pid, "tile_id": f"T{s_idx}", "season": "spring",
                         "labels": "|".join(LABELS[: 1 + i % 3]),
                         "B04_path": f"patches/{pid}/{pid}_B04.tif",
                         "B03_path": f"patches/{pid}/{pid}_B03.tif",
                         "B02_path": f"patches/{pid}/{pid}_B02.tif"})
        with open(split_dir / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return split_dir


@pytest.mark.parametrize("mode", ["head_only", "pooled_lora", "oracle_lora"])
def test_train_baseline_runs_one_epoch_on_cpu(tmp_path, mode):
    split_dir = _make_dataset(tmp_path)
    out = tmp_path / f"out_{mode}"
    cmd = [sys.executable, "-m", "src.training.train_baseline", "--mode", mode,
           "--split_dir", str(split_dir), "--output_dir", str(out), "--data_root", str(tmp_path),
           "--epochs", "1", "--batch_size", "4", "--num_workers", "0", "--device", "cpu"]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    rows = list(csv.DictReader(open(out / "metrics.csv", encoding="utf-8")))
    assert {r["split"] for r in rows} >= ({"oracle_train_unseen", "oracle_test_unseen"} if mode == "oracle_lora"
                                          else {"train_seen", "val_seen", "test_unseen"})
    assert (out / "best_checkpoint.pt").is_file() and (out / "last_checkpoint.pt").is_file()
    if mode != "head_only":
        assert "LoRA adapters inserted" in res.stdout
