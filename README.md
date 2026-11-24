# Glue

Data-preparation and baseline-training scaffolding for compositional-generalisation experiments on BigEarthNet-S2: do multi-label land-cover classifiers generalise to unseen combinations of Sentinel-2 tile and season?

**Status:** early-stage research utility. It contains data tooling, three baseline modes and NCI Gadi job scripts. **No results are published here.**

## What is implemented

| Component | File | What it does |
|---|---|---|
| Metadata index | `src/data/build_ben_s2_metadata.py`, `validate_ben_s2_metadata.py` | Scan a BigEarthNet-S2 copy into a patch-level table (tile, season, labels, band paths) and check it |
| Split construction | `src/data/recommend_ben_s2_split_candidates.py`, `create_ben_s2_first_split.py` | Choose held-out (tile, season) cells and write `train_seen` / `val_seen` / `test_unseen` plus `oracle_train_unseen` / `oracle_test_unseen` CSVs |
| Subset export | `src/data/export_ben_s2_subset.py` | Copy only the RGB bands (B04/B03/B02) of the selected patches, for transfer to HPC |
| Dataset / loader | `src/data/ben_s2_dataset.py`, `check_ben_s2_loader.py` | RGB tensors (reflectance clipped to [0, 10000] and scaled to [0, 1], resized to 224) and multi-hot labels; loader sanity check |
| Baselines | `src/training/train_baseline.py` | ResNet-50 with a new linear head, trained with AdamW + BCE in three modes: `head_only` (frozen backbone); `pooled_lora` (low-rank 1×1 conv adapters on `layer4`, trained on seen cells); `oracle_lora` (the same adapters trained on the unseen cells, as an upper reference). Reports macro mAP and micro/macro F1 |
| HPC | `nci/*.pbs`, `nci/setup_env.sh`, `nci/README_NCI.md` | PBS jobs for NCI Gadi (`gpuvolta`, 1 GPU, 8 CPUs, 64 GB) with placeholders for project, user and paths |

## Quick start

```bash
pip install -r requirements.txt          # plus a PyTorch build for your platform
python -m pytest tests                   # CPU smoke test on synthetic data (about 20 s)
python -m src.training.train_baseline --help
```

To train on real data, build the metadata and splits with the `src/data` scripts, then:

```bash
python -m src.training.train_baseline --mode pooled_lora --split_dir <splits> \
  --output_dir <out> --data_root <BigEarthNet-S2 or exported subset> --pretrained true
```

`--data_root` (or `BEN_S2_DATA_ROOT`) is only needed when the band paths in the split CSVs are not valid on the current machine. On NCI Gadi, replace `<PROJECT_CODE>` and `<USER>` in `nci/*.pbs` (see `nci/README_NCI.md`) and submit with `qsub`.

## Requirements

- [BigEarthNet-S2](https://bigearth.net/) (not included; see its own licence and terms).
- Python 3.9+, PyTorch/torchvision, pandas, pyarrow, tifffile, scikit-learn.
- A CUDA GPU is advisable for real runs; the smoke test runs on CPU.
- `--pretrained true` downloads torchvision's ImageNet ResNet-50 weights.

## Limitations

- No experiment results, figures or configuration of a finished study are included.
- In `oracle_lora` mode there is no validation split, so `best_checkpoint.pt` is selected on `oracle_test_unseen`. Treat oracle numbers as an optimistic upper reference.
- Inputs are not normalised with ImageNet statistics when `--pretrained true` is used.
- Only RGB bands are used.
- Tests cover a synthetic end-to-end run, not metric correctness on real data.

## Attribution

- BigEarthNet: G. Sumbul et al. The dataset is used under its own terms and is not redistributed.
- ResNet-50 from torchvision. The LoRA-style adapters are a small in-repo implementation (`LoRAConv2d`).
- Developed during PhD research at Deakin University. Licence: not yet declared.
