# GitHub repository

## Display name (recommended)

**Battery SOH Paper Reproduction**

## Current remote URL

| | |
|:---|:---|
| **Repository** | `MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--` |
| **Web** | https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION-- |
| **SSH** | `git@github.com:VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--.git` |

### Naming corrections applied in documentation

| Incorrect | Correct |
|:---|:---|
| PREDICATION | PREDICTION |
| SOH-RUL (public repo) | SOH only (RUL work in `local_archive/`) |
| `battery SOH predications` (local folder typo) | `battery_soh_prediction` recommended locally |

### Recommended GitHub rename (Settings → Repository name)

```
battery-soh-paper-reproduction
```

Alternatives: `hybrid-deep-learning-battery-soh-prediction`, `battery-soh-prediction-paper-reproduction`

After rename, update `git remote set-url origin` and README clone URLs.

## Clone and run

```powershell
git lfs install
git clone git@github.com:VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--
git lfs pull
pip install -r requirements.txt
python run_paper_experiment.py --require-real --cpu
```

## Commits (sole author)

```powershell
python scripts/git_commit_sole_author.py "your message here"
```
