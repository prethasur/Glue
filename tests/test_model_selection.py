"""Model-selection protocol: test data never selects a checkpoint; the selected weights are evaluated."""
import argparse
import csv
import json

import numpy as np
import torch

from src.training import train_baseline as tb


def _args(split_dir, out, root, mode, epochs=3):
    return argparse.Namespace(mode=mode, split_dir=split_dir, output_dir=out, data_root=root,
                              pretrained="false", pretrained_weights=None, epochs=epochs, batch_size=4,
                              lr=1e-3, seed=0, device="cpu", num_workers=0, lora_rank=2, lora_alpha=4.0,
                              lora_target_layers="layer4")


def _fake_run_epoch(log, val_maps, test_map_by_epoch_seen):
    """Replace run_epoch: record which split is evaluated, return scripted metrics.

    Validation mAP follows `val_maps`; test mAP is made *highest* at a different epoch so that
    any test-based selection would be detectable.
    """
    state = {"epoch": 0}

    def fake(model, loader, criterion, device, optimizer=None):
        split = loader.dataset.csv_path.stem
        fc = model.fc.weight.detach().clone()
        log.append((split, optimizer is not None, fc))
        if optimizer is not None:  # a "training" epoch: change the weights deterministically
            state["epoch"] += 1
            with torch.no_grad():
                model.fc.weight.fill_(float(state["epoch"]))
            m = 0.1
        elif split == "val_seen":
            m = val_maps[state["epoch"] - 1]
        else:
            m = test_map_by_epoch_seen.get(state["epoch"], 0.0)
        return {"loss": 1.0, "mAP": m, "micro_f1": 0.0, "macro_f1": 0.0, "n_classes_evaluated": 3}
    return fake


def test_val_selection_and_best_weights_used_for_test(synthetic_splits, monkeypatch):
    split_dir, root = synthetic_splits
    out = root / "out"
    log = []
    # val best at epoch 2; test would be best at epoch 3 (should be ignored)
    monkeypatch.setattr(tb, "run_epoch", _fake_run_epoch(log, val_maps=[0.2, 0.9, 0.5], test_map_by_epoch_seen={3: 0.99}))
    info = tb.train_baseline(_args(split_dir, out, root, "pooled_lora"))

    assert info["selection"] == {"policy": "best_val_mAP", "selection_split": "val_seen", "selected_epoch": 2,
                                 "selected_val_mAP": 0.9, "checkpoint_file": "best_checkpoint.pt"}
    splits_in_order = [s for s, _, _ in log]
    first_test = min(i for i, s in enumerate(splits_in_order) if "test" in s)
    assert all("test" not in s for s in splits_in_order[:first_test])
    assert all(s in ("train_seen", "val_seen") for s in splits_in_order[:first_test])  # training phase
    assert splits_in_order[first_test:] == ["test_unseen", "oracle_test_unseen"]       # evaluated once each
    # the held-out evaluation used epoch-2 weights (fc filled with 2.0), not the last epoch (3.0)
    for split, _, fc in log[first_test:]:
        assert torch.allclose(fc, torch.full_like(fc, 2.0)), split
    best = torch.load(out / "best_checkpoint.pt", map_location="cpu")
    last = torch.load(out / "last_checkpoint.pt", map_location="cpu")
    assert best["epoch"] == 2 and best["selection_split"] == "val_seen"
    assert torch.allclose(best["model_state_dict"]["fc.weight"], torch.full_like(best["model_state_dict"]["fc.weight"], 2.0))
    assert torch.allclose(last["model_state_dict"]["fc.weight"], torch.full_like(last["model_state_dict"]["fc.weight"], 3.0))
    rows = list(csv.DictReader(open(out / "metrics.csv", encoding="utf-8")))
    assert {r["checkpoint"] for r in rows if "test" in r["split"]} == {"best_val_mAP_epoch_2"}


def test_oracle_mode_never_selects_on_test(synthetic_splits, monkeypatch):
    split_dir, root = synthetic_splits
    out = root / "oracle"
    log = []
    monkeypatch.setattr(tb, "run_epoch", _fake_run_epoch(log, val_maps=[], test_map_by_epoch_seen={1: 0.99, 2: 0.1, 3: 0.2}))
    info = tb.train_baseline(_args(split_dir, out, root, "oracle_lora"))

    splits_in_order = [s for s, _, _ in log]
    assert splits_in_order == ["oracle_train_unseen"] * 3 + ["oracle_test_unseen"]   # test seen once, at the end
    assert info["selection"]["policy"] == "fixed_final_epoch" and info["selection"]["selected_epoch"] == 3
    assert not (out / "best_checkpoint.pt").exists()
    fc = log[-1][2]
    assert torch.allclose(fc, torch.full_like(fc, 3.0))  # final-epoch weights, despite test peaking at epoch 1
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["final_eval_splits"] == ["oracle_test_unseen"]


def test_fit_with_val_never_receives_test_loader():
    # fit() only accepts train/val loaders; held-out loaders are created afterwards in train_baseline.
    import inspect
    params = list(inspect.signature(tb.fit).parameters)
    assert "test_loader" not in params and params[:3] == ["model", "train_loader", "val_loader"]


def test_compute_metrics_excludes_absent_classes():
    targets = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float32)
    logits = torch.tensor([[3., -3., 0.], [-3., 3., 1.], [2., 1., -2.], [-3., -3., 4.]])
    m = tb.compute_metrics(logits, targets, loss=0.5)
    assert m["n_classes_evaluated"] == 2
    assert m["mAP"] == 1.0                      # perfect ranking on the two present classes
    assert np.isfinite(m["micro_f1"]) and np.isfinite(m["macro_f1"])


def test_compute_metrics_all_negative_split_is_nan_not_zero():
    m = tb.compute_metrics(torch.zeros(3, 2), torch.zeros(3, 2), loss=0.1)
    assert m["n_classes_evaluated"] == 0 and np.isnan(m["mAP"]) and np.isnan(m["macro_f1"])
