# Data and Git

## Raw data in Git (LFS)

After clone:

```powershell
git lfs install
git lfs pull
```

Tracked via `.gitattributes`:

- `data/**/*.mat`
- `data/**/*.xlsx`, `data/**/*.xls`

Approximate size: **~450 MB** (79 files).

## If LFS is unavailable

```powershell
python download_data.py --all
```

## What not to commit

| Path | Reason |
|:---|:---|
| `local_archive/` | MSc extension (local dissertation only) |
| `*.docx`, `MSc_Report*` | Final thesis report |
| `checkpoints/` | Regenerated during training |
| `results/*.log`, `validation_predictions.json` | Ephemeral |

## Paper-only entry

| Step | Command |
|:---|:---|
| Verify | `python scripts/verify_setup.py` |
| Train + evaluate | `python run_paper_experiment.py --require-real --cpu` |
| Figures | `python generate_figures.py` |
