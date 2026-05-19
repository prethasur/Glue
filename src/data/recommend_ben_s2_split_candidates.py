"""Recommend held-out tile-season cells for BigEarthNet-S2 split design."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MIN_HELDOUT_SAMPLES = 8000
MIN_TRAIN_CELLS = 3
MAX_TRAIN_CELLS = 6
TOP_CANDIDATES = 10


@dataclass(frozen=True)
class CellStats:
    tile_id: str
    season: str
    sample_count: int
    label_counts: Counter[str]

    @property
    def key(self) -> tuple[str, str]:
        return (self.tile_id, self.season)

    @property
    def labels(self) -> set[str]:
        return set(self.label_counts)


def split_labels(labels: str) -> list[str]:
    return [label.strip() for label in str(labels).split("|") if label.strip()]


def distribution(label_counts: Counter[str], label_vocab: list[str]) -> list[float]:
    total = sum(label_counts.values())
    if total == 0:
        return [0.0 for _ in label_vocab]
    return [label_counts[label] / total for label in label_vocab]


def kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(p_i * math.log2(p_i / q_i) for p_i, q_i in zip(p, q) if p_i > 0 and q_i > 0)


def jensen_shannon_divergence(
    first: Counter[str],
    second: Counter[str],
    label_vocab: list[str],
) -> float:
    p = distribution(first, label_vocab)
    q = distribution(second, label_vocab)
    midpoint = [(p_i + q_i) / 2 for p_i, q_i in zip(p, q)]
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def load_cell_stats(metadata_csv: Path) -> tuple[dict[tuple[str, str], CellStats], list[str]]:
    df = pd.read_csv(metadata_csv, usecols=["tile_id", "season", "labels"])
    all_labels = sorted({label for labels in df["labels"].fillna("") for label in split_labels(labels)})
    grouped: dict[tuple[str, str], CellStats] = {}

    for (tile_id, season), group in df.groupby(["tile_id", "season"], sort=False):
        label_counts: Counter[str] = Counter()
        for labels in group["labels"].fillna(""):
            label_counts.update(split_labels(labels))
        grouped[(str(tile_id), str(season))] = CellStats(
            tile_id=str(tile_id),
            season=str(season),
            sample_count=len(group),
            label_counts=label_counts,
        )
    return grouped, all_labels


def eligible_heldouts(cells: dict[tuple[str, str], CellStats]) -> list[CellStats]:
    tile_to_seasons: dict[str, set[str]] = {}
    season_to_tiles: dict[str, set[str]] = {}
    for cell in cells.values():
        tile_to_seasons.setdefault(cell.tile_id, set()).add(cell.season)
        season_to_tiles.setdefault(cell.season, set()).add(cell.tile_id)

    candidates = [
        cell
        for cell in cells.values()
        if cell.sample_count >= MIN_HELDOUT_SAMPLES
        and len(tile_to_seasons[cell.tile_id]) >= 2
        and len(season_to_tiles[cell.season]) >= 3
    ]
    return sorted(candidates, key=lambda cell: cell.sample_count, reverse=True)


def combined_label_counts(cells: list[CellStats]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for cell in cells:
        counts.update(cell.label_counts)
    return counts


def choose_train_cells(
    heldout: CellStats,
    cells: dict[tuple[str, str], CellStats],
    label_vocab: list[str],
) -> list[CellStats]:
    pool = [
        cell
        for cell in cells.values()
        if cell.key != heldout.key and cell.sample_count >= 3000
    ]
    pool.sort(key=lambda cell: cell.sample_count, reverse=True)

    selected: list[CellStats] = []
    remaining = pool[:120]
    heldout_labels = heldout.labels
    all_labels = set(label_vocab)

    def add_axis_anchor(candidates: list[CellStats]) -> None:
        if not candidates or len(selected) >= MAX_TRAIN_CELLS:
            return
        best_cell = min(
            candidates,
            key=lambda cell: (
                len(heldout_labels - cell.labels) * 100
                + jensen_shannon_divergence(cell.label_counts, heldout.label_counts, label_vocab)
                - cell.sample_count / 100_000
            ),
        )
        selected.append(best_cell)

    same_tile_other_seasons = [
        cell
        for cell in remaining
        if cell.tile_id == heldout.tile_id and cell.season != heldout.season
    ]
    same_season_other_tiles = [
        cell
        for cell in remaining
        if cell.season == heldout.season and cell.tile_id != heldout.tile_id
    ]
    add_axis_anchor(same_tile_other_seasons)
    add_axis_anchor([cell for cell in same_season_other_tiles if cell.key not in {item.key for item in selected}])
    selected_keys = {cell.key for cell in selected}
    remaining = [cell for cell in remaining if cell.key not in selected_keys]

    while len(selected) < MAX_TRAIN_CELLS and remaining:
        best_cell = None
        best_score = None
        for cell in remaining:
            trial = selected + [cell]
            train_counts = combined_label_counts(trial)
            missing_heldout = len(heldout_labels - set(train_counts))
            missing_all = len(all_labels - set(train_counts))
            jsd = jensen_shannon_divergence(train_counts, heldout.label_counts, label_vocab)
            sample_bonus = cell.sample_count / 100_000
            same_tile_bonus = 0.015 if cell.tile_id == heldout.tile_id else 0.0
            same_season_bonus = 0.01 if cell.season == heldout.season else 0.0
            score = missing_heldout * 100 + missing_all * 10 + jsd - sample_bonus - same_tile_bonus - same_season_bonus
            if best_score is None or score < best_score:
                best_score = score
                best_cell = cell
        if best_cell is None:
            break
        selected.append(best_cell)
        remaining = [cell for cell in remaining if cell.key != best_cell.key]

        train_labels = set(combined_label_counts(selected))
        if len(selected) >= MIN_TRAIN_CELLS and not (heldout_labels - train_labels) and not (all_labels - train_labels):
            break

    while len(selected) < MIN_TRAIN_CELLS and remaining:
        selected.append(remaining.pop(0))
    return selected


def recommend(metadata_csv: Path, output_csv: Path) -> pd.DataFrame:
    cells, label_vocab = load_cell_stats(metadata_csv)
    rows = []
    for rank, heldout in enumerate(eligible_heldouts(cells)[:TOP_CANDIDATES], start=1):
        train_cells = choose_train_cells(heldout, cells, label_vocab)
        train_counts = combined_label_counts(train_cells)
        missing_from_train = sorted(heldout.labels - set(train_counts))
        jsd = jensen_shannon_divergence(train_counts, heldout.label_counts, label_vocab)
        rows.append(
            {
                "rank": rank,
                "heldout_tile_id": heldout.tile_id,
                "heldout_season": heldout.season,
                "heldout_sample_count": heldout.sample_count,
                "suggested_train_cells": ";".join(f"{cell.tile_id}_{cell.season}" for cell in train_cells),
                "train_sample_count": sum(cell.sample_count for cell in train_cells),
                "train_cell_count": len(train_cells),
                "train_unique_labels": len(set(train_counts)),
                "heldout_unique_labels": len(heldout.labels),
                "labels_missing_from_train_relative_to_heldout": "|".join(missing_from_train),
                "label_js_divergence": round(jsd, 6),
            }
        )

    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata_csv",
        default=Path("outputs/ben_s2_clean_metadata.csv"),
        type=Path,
        help="Clean BigEarthNet-S2 metadata CSV.",
    )
    parser.add_argument(
        "--output_csv",
        default=Path("outputs/ben_s2_split_candidates.csv"),
        type=Path,
        help="CSV path for recommended candidate split cells.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_csv = args.metadata_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    if not metadata_csv.is_file():
        print(f"ERROR: metadata CSV not found: {metadata_csv}")
        return 2
    result = recommend(metadata_csv, output_csv)
    print(result.to_string(index=False))
    print(f"\nSaved: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
