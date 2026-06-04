# Data and Git LFS

## Clone

```powershell
git lfs install
git clone https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--
git lfs pull
```

## Datasets (~450 MB via LFS)

| Dataset | Path |
|:---|:---|
| NASA | `data/NASA/*.mat` |
| Oxford | `data/Oxford/Oxford_Battery_Degradation_Dataset_1.mat` |
| CALCE | `data/CALCE/**/*.xlsx` |

If LFS is unavailable: `python download_data.py --all`

## Not committed

- `local_archive/` — MSc extension (dissertation only)
- `checkpoints/`, `*.docx`, thesis reports
