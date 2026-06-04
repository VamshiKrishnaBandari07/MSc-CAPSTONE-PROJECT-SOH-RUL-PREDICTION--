# Academic & Portfolio Assessment

**Repository:** Battery SOH Paper Reproduction (Rahman et al., 2026)  
**Assessment date:** June 2026

---

## Scores (/100)

| Audience | Score | Rationale |
|:---|:---:|:---|
| **MSc dissertation** | **88** | Honest limitations, strong methodology alignment, Oxford metric match, professional docs |
| **Academic examiner / reproducibility** | **85** | Fixed seeds, LFS data, CV protocol; NASA RMSE gap documented |
| **Recruiter / ML engineer portfolio** | **86** | Clean GitHub story, PyTorch pipeline, real benchmarks, CI tests |
| **Overall publication-quality repo** | **87** | Paper-only scope; rename GitHub slug for full polish |

---

## Dimension breakdown

| Dimension | /10 |
|:---|:---:|
| Paper methodology fidelity | 9.0 |
| Documentation honesty | 9.5 |
| Code quality (post-cleanup) | 8.5 |
| Reproducibility | 8.5 |
| Repository hygiene | 9.0 |
| Numerical replication | 6.5 |
| Naming / branding | 7.0 |

---

## Supervisor-facing summary

This artefact implements the *Scientific Reports* (2026) hybrid SOH estimator with public NASA, Oxford, and CALCE data. Stratified 5-fold CV was used. Oxford SOH RMSE (**0.0215 ± 0.0050**) matches the published hybrid target within variance. NASA SOH RMSE (**0.0385**) aligns with the Transformer baseline, not the paper’s **0.021** hybrid claim — stated explicitly in README and JSON `reproducibility_note`.

---

## Recruiter-facing summary

End-to-end PyTorch research pipeline: feature engineering from electrochemical curves, temporal deep model (~0.39M params), cross-validation, automated figures, pytest CI, Git LFS datasets. Scope is reproducibility research, not product deployment.

---

## Files on GitHub vs local

See `docs/FINAL_FILE_MANIFEST.md` and `local_archive/LOCAL_ARCHIVE_CONTENTS.md`.
