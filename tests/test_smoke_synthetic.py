"""End-to-end CPU smoke test of all three training modes on synthetic data.

Run from the repository root:  python -m pytest tests
"""
import csv
import json
import subprocess
import sys

import pytest

from conftest import REPO


@pytest.mark.parametrize("mode", ["head_only", "pooled_lora", "oracle_lora"])
def test_train_baseline_runs_on_cpu(synthetic_splits, mode):
    split_dir, root = synthetic_splits
    out = root / f"out_{mode}"
    cmd = [sys.executable, "-m", "src.training.train_baseline", "--mode", mode,
           "--split_dir", str(split_dir), "--output_dir", str(out), "--data_root", str(root),
           "--pretrained", "false", "--epochs", "2", "--batch_size", "4", "--num_workers", "0",
           "--device", "cpu"]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-2000:]

    rows = list(csv.DictReader(open(out / "metrics.csv", encoding="utf-8")))
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["backbone_init"] == "random" and meta["protocol_version"] == 2
    assert (out / "last_checkpoint.pt").is_file()

    if mode == "oracle_lora":
        assert meta["selection"]["policy"] == "fixed_final_epoch"
        assert not (out / "best_checkpoint.pt").exists()
        assert [r["split"] for r in rows if r["split"] == "oracle_test_unseen"] == ["oracle_test_unseen"]
    else:
        assert meta["selection"]["selection_split"] == "val_seen"
        assert (out / "best_checkpoint.pt").is_file()
        held_out = [r for r in rows if r["split"] in ("test_unseen", "oracle_test_unseen")]
        assert len(held_out) == 2
        assert all(r["checkpoint"].startswith("best_val_mAP_epoch_") for r in held_out)
    if mode != "head_only":
        assert "LoRA adapters inserted" in res.stdout
