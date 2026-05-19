# Running BigEarthNet-S2 Experiments on NCI Gadi

This folder contains starter scripts for running the BigEarthNet-S2 loader checks and baseline jobs on NCI Gadi. Replace placeholders before use:

- `<PROJECT_CODE>`
- `<USER>`
- `<DATA_ROOT>`
- `<REPO_ROOT>`
- `<OUTPUT_ROOT>`

## Clone the Repository

Log in to Gadi, move to a project or scratch location, then clone the repo:

```bash
ssh <USER>@gadi.nci.org.au
cd /scratch/<PROJECT_CODE>/<USER>
git clone <YOUR_REPO_URL> Glue
cd Glue
```

If you copy the project manually, set `<REPO_ROOT>` to that directory.

## Data Location

Place BigEarthNet-S2 somewhere on project or scratch storage, for example:

```text
<DATA_ROOT>/BigEarthNet-S2/
```

The expected dataset folder should contain local patch folders and `metadata.parquet`:

```text
<DATA_ROOT>/BigEarthNet-S2/metadata.parquet
<DATA_ROOT>/BigEarthNet-S2/S2A_MSIL2A_.../
```

The prepared split CSVs should live under:

```text
<OUTPUT_ROOT>/ben_s2_splits_first/
```

If you generate metadata and splits on Gadi, write outputs to project or scratch storage, not `$HOME`.

## Export a Portable First Subset Locally

The first split CSVs created on Windows contain absolute local paths. Before copying to Gadi, export a portable subset:

```powershell
python -m src.data.export_ben_s2_subset --split_dir outputs/ben_s2_splits_first --subset_root outputs/ben_s2_first_subset
```

This creates:

```text
outputs/ben_s2_first_subset/patches/
outputs/ben_s2_first_subset/splits/
```

The rewritten split CSVs use relative paths such as:

```text
patches/<patch_id>/<patch_id>_B02.tif
```

Check the portable subset locally:

```powershell
python -m src.data.check_ben_s2_loader --split_dir outputs/ben_s2_first_subset/splits --data_root outputs/ben_s2_first_subset --batch_size 8
```

## Copy the Portable Subset to Gadi

Use `rsync` from your local machine, replacing placeholders:

```bash
rsync -av --progress outputs/ben_s2_first_subset/ <USER>@gadi.nci.org.au:<OUTPUT_ROOT>/ben_s2_first_subset/
```

Also copy or clone the repository itself to `<REPO_ROOT>`. On Gadi, the portable split directory is:

```text
<OUTPUT_ROOT>/ben_s2_first_subset/splits
```

and the data root for patch resolution is:

```text
<OUTPUT_ROOT>/ben_s2_first_subset
```

## Python Environment

From `<REPO_ROOT>`:

```bash
bash nci/setup_env.sh
source .venvs/Glue/bin/activate
```

The setup script uses generic module commands. NCI module names change over time, so edit the commented `module load` lines if your project uses a different Python, CUDA, or conda module.

## Submit PBS Jobs

Create logs first if submitting from the repo root:

```bash
mkdir -p logs
```

Submit the loader check:

```bash
qsub nci/run_loader_check.pbs
```

Submit the head-only baseline:

```bash
qsub nci/run_train_head.pbs
```

You can override script variables at submission time:

```bash
qsub -v SUBSET_ROOT=<OUTPUT_ROOT>/ben_s2_first_subset,OUTPUT_DIR=<OUTPUT_ROOT>/head_only_seed0 nci/run_train_head.pbs
```

## Check Jobs

```bash
qstat -u $USER
```

## Inspect Logs

PBS stdout and stderr are written to `logs/` by the templates:

```bash
ls -lh logs/
tail -n 100 logs/*.o*
tail -n 100 logs/*.e*
```

If a job exits quickly, inspect the `.e` file first. It usually contains missing module, missing path, or environment activation errors.

## Notes

- `run_train_head.pbs` is the first runnable training job.
- `run_train_pooled_lora.pbs` and `run_train_oracle_lora.pbs` are templates for the next phase. The Python entrypoint reserves these modes, but LoRA training is intentionally not implemented yet.
- Keep dataset files and large outputs on project or scratch storage. Avoid writing large files to `$HOME`.
