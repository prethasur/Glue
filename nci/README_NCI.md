# Running Glue on NCI Gadi

These are PBS **templates**. They encode the resource requests used previously (`gpuvolta`, 1 GPU) and read all site-specific paths from environment variables, but this revision has **not been re-run on Gadi**. Replace placeholders and check module names for your project before use.

## Placeholders and variables

| Where | Name | Meaning |
|---|---|---|
| `#PBS` directives | `<PROJECT_CODE>` | Your NCI project. Edit the directives or override them on the command line: `qsub -P <PROJECT_CODE> -l storage=gdata/<PROJECT_CODE>+scratch/<PROJECT_CODE> ...` |
| `qsub -v` | `GLUE_ROOT` | Repository checkout on Gadi |
| `qsub -v` | `SPLIT_DIR` | Directory containing the split CSVs |
| `qsub -v` | `DATA_ROOT` | Root used to resolve band paths (the portable subset root) |
| `qsub -v` | `OUTPUT_DIR` | Output directory for one run (training jobs) |
| `qsub -v` | `PRETRAINED` | `true` or `false`. Required. `false` trains on a **random frozen backbone** (a negative control) |
| `qsub -v` (optional) | `PRETRAINED_WEIGHTS` | Local torchvision ResNet-50 `state_dict`. Gadi compute nodes have no internet access, so download the weights once on a login node and pass the file |
| `qsub -v` (optional) | `SEED`, `EPOCHS`, `BATCH_SIZE`, `LR`, `LORA_RANK`, `LORA_ALPHA`, `LORA_TARGET_LAYERS`, `NUM_WORKERS`, `VENV`, `TORCH_MODULE` | Defaults are in `train_common.sh` |

The jobs stop immediately with a clear message if a required variable is missing.

## 1. Prepare a portable data subset (local machine)

Split CSVs created by `create_ben_s2_first_split` contain absolute local paths. Export a portable copy with relative paths:

```bash
python -m src.data.export_ben_s2_subset --split_dir outputs/ben_s2_splits_first \
  --subset_root outputs/ben_s2_first_subset --rgb_only
python -m src.data.check_ben_s2_loader --split_dir outputs/ben_s2_first_subset/splits \
  --data_root outputs/ben_s2_first_subset --batch_size 8
```

This creates `patches/<patch_id>/<patch_id>_B0{2,3,4}.tif` and `splits/*.csv`. Copy it to project or scratch storage (not `$HOME`):

```bash
rsync -av outputs/ben_s2_first_subset/ <USER>@gadi.nci.org.au:/scratch/<PROJECT_CODE>/<USER>/ben_s2_first_subset/
```

## 2. Environment (Gadi login node)

```bash
cd /scratch/<PROJECT_CODE>/<USER>/Glue
bash nci/setup_env.sh        # user-level venv on top of the pytorch module; edit the module name if needed
```

`setup_env.sh` loads `pytorch/1.10.0` and installs `torchvision==0.11.1` plus the data dependencies into `.venvs/Glue`. Module names change over time, so check `module avail pytorch`.

Optional: fetch the ImageNet ResNet-50 weights on the login node for offline use:

```bash
source .venvs/Glue/bin/activate
python -c "import torch, torchvision; m = torchvision.models.resnet50(pretrained=True); torch.save(m.state_dict(), 'resnet50_imagenet.pth')"
```

## 3. Submit

```bash
G=/scratch/<PROJECT_CODE>/<USER>
COMMON="GLUE_ROOT=$G/Glue,SPLIT_DIR=$G/ben_s2_first_subset/splits,DATA_ROOT=$G/ben_s2_first_subset"

qsub -v "$COMMON" nci/run_loader_check.pbs
qsub -v "$COMMON,OUTPUT_DIR=$G/glue_outputs/head_only_seed0,PRETRAINED=true,PRETRAINED_WEIGHTS=$G/Glue/resnet50_imagenet.pth" nci/run_train_head.pbs
qsub -v "$COMMON,OUTPUT_DIR=$G/glue_outputs/pooled_lora_seed0,PRETRAINED=true,PRETRAINED_WEIGHTS=$G/Glue/resnet50_imagenet.pth" nci/run_train_pooled_lora.pbs
qsub -v "$COMMON,OUTPUT_DIR=$G/glue_outputs/oracle_lora_seed0,PRETRAINED=true,PRETRAINED_WEIGHTS=$G/Glue/resnet50_imagenet.pth" nci/run_train_oracle_lora.pbs
```

All three training modes (`head_only`, `pooled_lora`, `oracle_lora`) are implemented in `src/training/train_baseline.py`. `run_train_*.pbs` set `MODE` and a default learning rate, then run `nci/train_common.sh`.

## 4. Monitor and collect

```bash
qstat -u $USER
ls *.o*                                  # PBS output: <job name>.o<job id>, in the submission directory
cat $G/glue_outputs/head_only_seed0/run_metadata.json
```

Each run writes `metrics.csv`, `run_metadata.json`, `last_checkpoint.pt`, and `best_checkpoint.pt` (non-oracle modes only).

## Notes

- Results produced before this revision used `--pretrained false`, selected the oracle checkpoint on the test split, evaluated the last (not the validation-selected) epoch, and averaged AP over absent classes. They are not comparable with results from the current code (`protocol_version: 2` in `run_metadata.json`).
- Earlier templates set `#PBS -o .../$PBS_JOBID.log`. PBS does not expand variables in directives, so runs overwrote a file literally named `$PBS_JOBID.log`. The templates now use the PBS default output file.
