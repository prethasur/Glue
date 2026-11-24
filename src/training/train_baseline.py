"""Train ResNet-50 baselines for the BigEarthNet-S2 tile-season split experiments.

Modes
-----
head_only    Frozen ResNet-50 backbone, new linear head. Trained on train_seen,
             checkpoint selected on val_seen, evaluated once on test_unseen and
             oracle_test_unseen.
pooled_lora  As head_only, plus trainable low-rank convolutional residual adapters
             (LoRAConv2d) on the target layers (default: layer4).
oracle_lora  Same adapters, trained on oracle_train_unseen (half of the held-out
             cell). There is no oracle validation split, so training runs for a
             fixed number of epochs and the final model is evaluated once on
             oracle_test_unseen. No test metric is used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from src.data.ben_s2_dataset import (
    BigEarthNetS2RGBDataset,
    build_label_to_index,
    load_label_to_index,
    save_label_to_index,
)


MetricRow = Dict[str, Union[float, int, str]]
METRIC_FIELDS = ["epoch", "split", "checkpoint", "loss", "mAP", "micro_f1", "macro_f1", "n_classes_evaluated"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str_to_bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_targets(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_resnet50(pretrained: bool, weights_path: Optional[Path] = None) -> Tuple[nn.Module, str]:
    """Return (model, init_description). Never falls back silently to random weights.

    pretrained=False                -> random initialisation.
    pretrained=True, weights_path   -> load a local torchvision ResNet-50 state_dict (strict).
    pretrained=True, no weights_path-> torchvision ImageNet weights (may download); errors propagate.
    """
    if not pretrained:
        if weights_path is not None:
            raise ValueError("--pretrained_weights was given but --pretrained is false.")
        try:
            model = models.resnet50(weights=None)
        except TypeError:  # torchvision < 0.13
            model = models.resnet50(pretrained=False)
        return model, "random"

    if weights_path is not None:
        try:
            model = models.resnet50(weights=None)
        except TypeError:
            model = models.resnet50(pretrained=False)
        try:  # tensors only (no arbitrary pickled objects) where torch supports it
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 1.13 has no weights_only
            state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        return model, f"pretrained:file:{Path(weights_path).name}"

    try:
        weights_enum = getattr(models, "ResNet50_Weights")
    except AttributeError:  # torchvision < 0.13: old API
        try:
            return models.resnet50(pretrained=True), "pretrained:torchvision-imagenet(legacy-api)"
        except Exception as exc:
            raise RuntimeError(
                "Could not load ImageNet-pretrained ResNet-50 weights. On machines without internet "
                "access, download them once and pass --pretrained_weights <file>."
            ) from exc
    try:
        return models.resnet50(weights=weights_enum.DEFAULT), f"pretrained:torchvision:{weights_enum.DEFAULT}"
    except Exception as exc:
        raise RuntimeError(
            "Could not load ImageNet-pretrained ResNet-50 weights. On machines without internet "
            "access, download them once and pass --pretrained_weights <file>."
        ) from exc


class LoRAConv2d(nn.Module):
    """Frozen Conv2d plus a trainable low-rank convolutional residual adapter.

    y = base(x) + (alpha / rank) * up(down(x)), where down is a 1x1 conv to `rank`
    channels with the base conv's stride, and up is a 1x1 conv back to the base output
    channels. `up` is zero-initialised, so the wrapped layer initially reproduces the
    frozen base layer exactly. This is a LoRA-style adapter for convolutions; it is not
    a low-rank factorisation of the k x k base kernel.
    """

    def __init__(self, base_conv: nn.Conv2d, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        k_h, k_w = base_conv.kernel_size
        if base_conv.groups != 1 or base_conv.dilation != (1, 1) or base_conv.padding != ((k_h - 1) // 2, (k_w - 1) // 2):
            raise ValueError(
                "LoRAConv2d supports groups=1, dilation=1 and 'same'-style padding only; "
                f"got groups={base_conv.groups}, dilation={base_conv.dilation}, padding={base_conv.padding}."
            )
        self.base_conv = base_conv
        for parameter in self.base_conv.parameters():
            parameter.requires_grad = False

        self.down = nn.Conv2d(base_conv.in_channels, rank, kernel_size=1, stride=base_conv.stride, bias=False)
        self.up = nn.Conv2d(rank, base_conv.out_channels, kernel_size=1, bias=False)
        self.scaling = alpha / float(rank)
        nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_conv(x) + self.up(self.down(x)) * self.scaling


def module_matches_target(module_name: str, targets: List[str]) -> bool:
    return any(module_name == target or module_name.startswith(f"{target}.") for target in targets)


def set_nested_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def add_lora_adapters(model: nn.Module, rank: int, alpha: float, target_layers: str) -> List[str]:
    targets = parse_targets(target_layers)
    replaced: List[str] = []
    for module_name, module in list(model.named_modules()):
        if not module_name or not isinstance(module, nn.Conv2d):
            continue
        if not module_matches_target(module_name, targets):
            continue
        set_nested_module(model, module_name, LoRAConv2d(module, rank=rank, alpha=alpha))
        replaced.append(module_name)
    if not replaced:
        raise ValueError(f"No Conv2d modules matched --lora_target_layers={target_layers!r}")
    return replaced


def freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def replace_classifier(model: nn.Module, num_labels: int) -> None:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)


def build_model(args: argparse.Namespace, num_labels: int) -> Tuple[nn.Module, str, List[str]]:
    weights_path = Path(args.pretrained_weights).expanduser() if args.pretrained_weights else None
    model, init = build_resnet50(pretrained=str_to_bool(args.pretrained), weights_path=weights_path)
    freeze_all_parameters(model)
    replaced: List[str] = []
    if args.mode in {"pooled_lora", "oracle_lora"}:
        replaced = add_lora_adapters(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                     target_layers=args.lora_target_layers)
        print(f"LoRA adapters inserted into {len(replaced)} Conv2d layer(s): {', '.join(replaced)}")
    replace_classifier(model, num_labels)
    return model, init, replaced


def set_train_mode(model: nn.Module, training: bool) -> None:
    """Train mode for trainable modules only; BatchNorm layers stay in eval mode.

    The backbone is frozen, so its BatchNorm running statistics must not be updated by
    the training data either (they would otherwise adapt even with requires_grad=False).
    """
    model.train(training)
    if training:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def make_loader(split_dir: Path, split_name: str, label_to_index: Dict[str, int], batch_size: int,
                shuffle: bool, num_workers: int, data_root: Optional[Path]) -> DataLoader:
    dataset = BigEarthNetS2RGBDataset(split_dir / f"{split_name}.csv", label_to_index, data_root=data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, loss: float) -> Dict[str, float]:
    """Multi-label metrics. Macro averages use only classes with >=1 positive in `targets`.

    Classes absent from the evaluated split have undefined AP (sklearn would score them 0),
    so they are excluded; `n_classes_evaluated` records how many classes were averaged.
    Predictions use a fixed 0.5 probability threshold for F1.
    """
    probabilities = torch.sigmoid(logits).cpu().numpy()
    y_true = targets.cpu().numpy().astype(np.int32)
    y_pred = (probabilities >= 0.5).astype(np.int32)
    present = np.flatnonzero(y_true.sum(axis=0) > 0)
    if present.size == 0:
        mean_ap = float("nan")
        macro_f1 = float("nan")
    else:
        mean_ap = float(average_precision_score(y_true[:, present], probabilities[:, present], average="macro"))
        macro_f1 = float(f1_score(y_true[:, present], y_pred[:, present], average="macro", zero_division=0))
    return {
        "loss": loss,
        "mAP": mean_ap,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": macro_f1,
        "n_classes_evaluated": int(present.size),
    }


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
              optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, float]:
    is_train = optimizer is not None
    set_train_mode(model, is_train)
    total_loss = 0.0
    total_examples = 0
    logits_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []

    for images, labels, _metadata in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        logits_chunks.append(logits.detach().cpu())
        target_chunks.append(labels.detach().cpu())

    if total_examples == 0:
        raise ValueError("Empty data loader: the split CSV has no rows.")
    mean_loss = total_loss / total_examples
    return compute_metrics(torch.cat(logits_chunks), torch.cat(target_chunks), mean_loss)


def write_metrics(path: Path, rows: List[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def split_plan(mode: str) -> Tuple[str, Optional[str], List[str]]:
    """Return (train split, selection split or None, final evaluation splits)."""
    if mode in {"head_only", "pooled_lora"}:
        # oracle_test_unseen is a subset of test_unseen; reporting it allows a like-for-like
        # comparison with oracle_lora.
        return "train_seen", "val_seen", ["test_unseen", "oracle_test_unseen"]
    if mode == "oracle_lora":
        return "oracle_train_unseen", None, ["oracle_test_unseen"]
    raise ValueError(f"Unsupported mode: {mode}")


def fit(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader],
        epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device,
        train_split: str, val_split: Optional[str], output_dir: Path, ckpt_extra: dict,
        epoch_fn: Callable = None) -> Tuple[List[MetricRow], Dict[str, object]]:
    """Train for `epochs`. Select a checkpoint on `val_loader` only (never on test data).

    Returns metric rows and a selection record. When val_loader is None, the final epoch is
    used (fixed-epoch policy). The selected weights are loaded into `model` before returning.
    """
    epoch_fn = epoch_fn or run_epoch
    rows: List[MetricRow] = []
    best_map, best_epoch, best_state = -float("inf"), None, None
    for epoch in range(1, epochs + 1):
        train_metrics = epoch_fn(model, train_loader, criterion, device, optimizer)
        rows.append({"epoch": epoch, "split": train_split, "checkpoint": f"epoch_{epoch}", **train_metrics})
        if val_loader is not None:
            val_metrics = epoch_fn(model, val_loader, criterion, device)
            rows.append({"epoch": epoch, "split": val_split, "checkpoint": f"epoch_{epoch}", **val_metrics})
            if val_metrics["mAP"] > best_map:  # NaN never compares greater
                best_map, best_epoch = val_metrics["mAP"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"epoch": epoch, "model_state_dict": best_state, "selection_split": val_split,
                            "selection_metrics": val_metrics, **ckpt_extra}, output_dir / "best_checkpoint.pt")
        write_metrics(output_dir / "metrics.csv", rows)

    torch.save({"epoch": epochs, "model_state_dict": model.state_dict(), **ckpt_extra},
               output_dir / "last_checkpoint.pt")

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        selection = {"policy": "best_val_mAP", "selection_split": val_split, "selected_epoch": best_epoch,
                     "selected_val_mAP": best_map, "checkpoint_file": "best_checkpoint.pt"}
    else:
        if val_loader is not None:
            print("WARNING: validation mAP was undefined in every epoch; using the final epoch.")
        selection = {"policy": "fixed_final_epoch", "selection_split": None, "selected_epoch": epochs,
                     "checkpoint_file": "last_checkpoint.pt"}
    return rows, selection


def train_baseline(args: argparse.Namespace) -> Dict[str, object]:
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    output_dir.mkdir(parents=True, exist_ok=True)

    label_path = split_dir / "label_to_index.json"
    if label_path.is_file():
        label_to_index = load_label_to_index(label_path)
    else:
        label_to_index = build_label_to_index(split_dir)
        save_label_to_index(label_to_index, label_path)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_split, val_split, eval_splits = split_plan(args.mode)

    set_seed(args.seed)
    model, init, adapted = build_model(args, num_labels=len(label_to_index))
    model = model.to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    print(f"mode: {args.mode} | backbone init: {init} | trainable parameters: {trainable_count}")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = lambda split, shuffle: make_loader(split_dir, split, label_to_index, args.batch_size, shuffle,
                                                args.num_workers, data_root)
    train_loader = loader(train_split, True)
    val_loader = loader(val_split, False) if val_split is not None else None

    ckpt_extra = {"label_to_index": label_to_index, "args": {k: str(v) for k, v in vars(args).items()},
                  "backbone_init": init}
    rows, selection = fit(model, train_loader, val_loader, args.epochs, optimizer, criterion, device,
                          train_split, val_split, output_dir, ckpt_extra)

    # Held-out evaluation: once, after training, with the selected weights.
    tag = f"{selection['policy']}_epoch_{selection['selected_epoch']}"
    for split in eval_splits:
        metrics = run_epoch(model, loader(split, False), criterion, device)
        row = {"epoch": selection["selected_epoch"], "split": split, "checkpoint": tag, **metrics}
        rows.append(row)
        print(row)
    write_metrics(output_dir / "metrics.csv", rows)

    run_info = {
        "mode": args.mode, "backbone_init": init, "trainable_parameters": trainable_count,
        "lora_layers": adapted, "train_split": train_split, "selection": selection,
        "final_eval_splits": eval_splits, "num_labels": len(label_to_index), "seed": args.seed,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "torch": torch.__version__, "torchvision": getattr(__import__("torchvision"), "__version__", "?"),
        "protocol_version": 2,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Saved metrics, checkpoints and run_metadata.json to {output_dir}")
    return run_info


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["head_only", "pooled_lora", "oracle_lora"])
    parser.add_argument("--split_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--data_root", default=None, type=Path, help="Optional BigEarthNet-S2 root or exported subset root.")
    parser.add_argument("--pretrained", required=True, choices=["true", "false"],
                        help="Backbone initialisation. 'false' = random frozen backbone (a negative control, "
                             "not a pretrained-feature baseline). Must be given explicitly.")
    parser.add_argument("--pretrained_weights", default=None, type=Path,
                        help="Local torchvision ResNet-50 state_dict (for offline nodes); requires --pretrained true.")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=8.0, type=float)
    parser.add_argument("--lora_target_layers", default="layer4", help="Comma-separated ResNet module prefixes to adapt.")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.pretrained_weights is not None and args.pretrained != "true":
        parser.error("--pretrained_weights requires --pretrained true")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    train_baseline(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
