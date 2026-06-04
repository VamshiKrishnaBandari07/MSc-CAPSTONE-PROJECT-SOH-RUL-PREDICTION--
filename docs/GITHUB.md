# GitHub Repository Notes

**Author:** Vamshi Krishna Bandari  
**Programme:** MSc Artificial Intelligence, University of Roehampton (UK)

---

## Canonical repository name

The repository should be named:

**`MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION`**

(Legacy name `MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICATION-` contained a spelling typo and trailing hyphen.)

### Rename on GitHub (one-time, repository owner)

1. Open **Settings → General → Repository name**
2. Set name to: `MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION`
3. Click **Rename**

GitHub redirects the old URL automatically. Update the CI badge in `README.md` after rename if desired.

### Update local remote after rename

```powershell
git remote set-url origin git@github.com:VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION.git
git remote -v
```

---

## Clone URL (canonical)

```bash
git lfs install
git clone git@github.com:VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION
git lfs pull
```

HTTPS:

```bash
git clone https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION.git
```

---

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` and `verify_setup.py` on every push to `main`.

---

*MSc Capstone software artefact — thesis report (`.docx`) is submitted locally, not via GitHub.*
