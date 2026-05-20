"""Train baseline models for BigEarthNet-S2 split experiments."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


def build_resnet50(pretrained: bool) -> nn.Module:
    if pretrained:
        try:
            weights_enum = getattr(models, "ResNet50_Weights")
            return models.resnet50(weights=weights_enum.DEFAULT)
        except Exception as exc:
            print(f"WARNING: failed to load pretrained ResNet-50 with new torchvision API: {exc}")
        try:
            return models.resnet50(pretrained=True)
        except Exception as exc:
            print(f"WARNING: failed to load pretrained ResNet-50 with old torchvision API: {exc}")
            print("WARNING: falling back to pretrained=False.")

    try:
        return models.resnet50(weights=None)
    except TypeError:
        return models.resnet50(pretrained=False)


class LoRAConv2d(nn.Module):
    """Frozen Conv2d plus a trainable low-rank 1x1 residual adapter."""

    def __init__(self, base_conv: nn.Conv2d, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_conv = base_conv
        for parameter in self.base_conv.parameters():
            parameter.requires_grad = False

        self.down = nn.Conv2d(
            base_conv.in_channels,
            rank,
            kernel_size=1,
            stride=base_conv.stride,
            bias=False,
        )
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


def build_model(args: argparse.Namespace, num_labels: int) -> nn.Module:
    model = build_resnet50(pretrained=str_to_bool(args.pretrained))
    freeze_all_parameters(model)
    if args.mode in {"pooled_lora", "oracle_lora"}:
        replaced = add_lora_adapters(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_layers=args.lora_target_layers,
        )
        print(f"LoRA adapters inserted into {len(replaced)} Conv2d layer(s): {', '.join(replaced)}")
    replace_classifier(model, num_labels)
    return model


def make_loader(
    split_dir: Path,
    split_name: str,
    label_to_index: Dict[str, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    data_root: Optional[Path],
) -> DataLoader:
    dataset = BigEarthNetS2RGBDataset(split_dir / f"{split_name}.csv", label_to_index, data_root=data_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, loss: float) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits).cpu().numpy()
    y_true = targets.cpu().numpy()
    y_pred = (probabilities >= 0.5).astype(np.int32)
    try:
        mean_ap = float(average_precision_score(y_true, probabilities, average="macro"))
    except ValueError:
        mean_ap = float("nan")
    return {
        "loss": loss,
        "mAP": mean_ap,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
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

    mean_loss = total_loss / max(total_examples, 1)
    return compute_metrics(torch.cat(logits_chunks), torch.cat(target_chunks), mean_loss)


def write_metrics(path: Path, rows: List[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["epoch", "split", "loss", "mAP", "micro_f1", "macro_f1"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_plan(mode: str) -> Tuple[str, Optional[str], str]:
    if mode in {"head_only", "pooled_lora"}:
        return "train_seen", "val_seen", "test_unseen"
    if mode == "oracle_lora":
        return "oracle_train_unseen", None, "oracle_test_unseen"
    raise ValueError(f"Unsupported mode: {mode}")


def train_baseline(args: argparse.Namespace) -> None:
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
    train_split, val_split, test_split = split_plan(args.mode)

    set_seed(args.seed)
    model = build_model(args, num_labels=len(label_to_index)).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(f"mode: {args.mode}")
    print(f"trainable parameters: {trainable_count}")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = make_loader(split_dir, train_split, label_to_index, args.batch_size, True, args.num_workers, data_root)
    val_loader = (
        make_loader(split_dir, val_split, label_to_index, args.batch_size, False, args.num_workers, data_root)
        if val_split is not None
        else None
    )
    test_loader = make_loader(split_dir, test_split, label_to_index, args.batch_size, False, args.num_workers, data_root)

    metrics_rows: List[MetricRow] = []
    best_eval_map = -1.0
    best_eval_name = val_split if val_split is not None else test_split

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        epoch_metrics: List[Tuple[str, Dict[str, float]]] = [(train_split, train_metrics)]
        if val_loader is not None:
            epoch_metrics.append((val_split, run_epoch(model, val_loader, criterion, device)))
        else:
            epoch_metrics.append((test_split, run_epoch(model, test_loader, criterion, device)))

        for split_name, metrics in epoch_metrics:
            row: MetricRow = {"epoch": epoch, "split": split_name}
            row.update(metrics)
            metrics_rows.append(row)
            print(row)

        eval_metrics = epoch_metrics[-1][1]
        if eval_metrics["mAP"] > best_eval_map:
            best_eval_map = eval_metrics["mAP"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "label_to_index": label_to_index,
                    "args": vars(args),
                    "eval_split": best_eval_name,
                    "eval_metrics": eval_metrics,
                },
                output_dir / "best_checkpoint.pt",
            )
        write_metrics(output_dir / "metrics.csv", metrics_rows)

    if val_loader is not None:
        test_metrics = run_epoch(model, test_loader, criterion, device)
        row = {"epoch": args.epochs, "split": test_split}
        row.update(test_metrics)
        metrics_rows.append(row)
        print(row)

    write_metrics(output_dir / "metrics.csv", metrics_rows)
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "args": vars(args),
        },
        output_dir / "last_checkpoint.pt",
    )
    print(f"Saved metrics and checkpoints to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["head_only", "pooled_lora", "oracle_lora"])
    parser.add_argument("--split_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--data_root", default=None, type=Path, help="Optional BigEarthNet-S2 root or exported subset root.")
    parser.add_argument("--pretrained", default="false", choices=["true", "false"])
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=8.0, type=float)
    parser.add_argument("--lora_target_layers", default="layer4", help="Comma-separated ResNet module prefixes to adapt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
