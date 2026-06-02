# MSc Artificial Intelligence Supervisor Audit Report

Repository: Battery SOH/RUL diagnostics with paper-aligned experiment package  
Review role: Senior UK MSc AI supervisor and GitHub reviewer  
Referenced paper: *Deep learning-based battery health prediction for enhancing electric vehicle performance*, Scientific Reports, DOI `10.1038/s41598-026-39911-8`

---

## Executive judgement

This repository is a credible MSc AI engineering prototype with a clear direction: reproduce a CNN-TCN-LSTM-Attention battery SOH model and extend it with a joint SOH/RUL physics-informed loss. The strongest part is the `paper_exp/` package, which now provides a paper-first workflow, Kaggle dataset preparation, training, and structured comparison reports.

However, as a final MSc dissertation submission, the repository still requires a stronger evidence bundle. The current implementation is best described as an **architecture-faithful reproduction attempt and thesis extension**, not a fully verified reproduction of the paper's reported metrics. The code can run, but the repository does not commit the real datasets, trained weights, full experimental logs, ablation tables, or a test suite.

**Supervisor grade for repository artefact:** **Merit**  
**Route to Distinction:** Add frozen results, ablations, tests, methodology figures, trained checkpoints, and a written dissertation chapter that critically explains deviations from the paper.

---

## 1. Repository structure review

### Strengths

- `paper_exp/` is now a coherent experiment package:
  - `prepare_data.py` for dataset conversion;
  - `train.py` for the paper-style SOH model;
  - `modified_experiment.py` for the existing SOH/RUL extension;
  - `compare_results.py` for JSON/Markdown reports;
  - `run_comparison.py` for paper-first orchestration.
- `data/*/PLACE_DATA_HERE.txt` guides are present for NASA, Oxford, CALCE, and KaggleSDG7.
- Generated outputs and downloaded data are ignored, preventing accidental large commits.
- Root scripts are still usable for quick synthetic demonstrations.

### Weaknesses

- No committed `tests/` directory.
- No `LICENSE`, `CITATION.cff`, `environment.yml`, `Dockerfile`, or CI workflow.
- No committed trained model checkpoints.
- No committed full-run `metrics.json` or experiment logs.
- No dissertation PDF, results chapter, ethics/data provenance note, or appendix artefacts.
- Root `model_paper.py` / `train_paper.py` are legacy demos and differ from the canonical `paper_exp/` implementation.

### Reproducibility verdict

Reproducibility is **partial**:

- A smoke workflow can run from a fresh clone.
- Real paper-level reproduction still depends on external datasets and long training.
- Full paper metrics are not reproducible from the repository alone.

---

## 2. Code review

### Good practice

- PyTorch modules are readable and separated by purpose.
- `paper_exp/train.py` uses JSON metrics export, early stopping, scheduling, and explicit paper targets.
- `paper_exp/preprocess.py` performs Savitzky-Golay smoothing, derivative extraction, voltage-grid alignment, and safe finite-value scaling.
- `paper_exp/prepare_data.py` handles multiple raw formats (`.mat`, `.csv`, `.txt`, `.xls`, `.xlsx`) and Kaggle download/manual placement workflows.
- `paper_exp/run_comparison.py` now enforces the intended order: paper experiment first, modified experiment second, comparison last.

### Bugs and risks found

| Priority | Issue | Impact | Current status |
| --- | --- | --- | --- |
| High | Benchmarking used requested `seq_len` rather than loaded processed tensor length | Latency could be measured on a different shape from training data | Fixed in `paper_exp/train.py` |
| High | Smoke comparison could fail without real Kaggle processed data | Poor fresh-clone reproducibility | Fixed: smoke mode creates demo Kaggle-style data if needed |
| High | Root `preprocess.py` never loads real files | Root results are synthetic by default | Documented limitation |
| High | Root `train.py` calls a single 80/20 split "cross-validation" | Methodological overclaim | Documented limitation; `paper_exp/` has the stronger workflow |
| Medium | Three paper-like paths exist (`paper_exp`, root `*_paper.py`, README claims) | Examiner confusion | README now identifies `paper_exp/` as canonical |
| Medium | No saved checkpoints | Hard to inspect trained models | Still missing |
| Medium | No automated parser tests | Dataset converter could regress | Still missing |
| Low | PyTorch deprecated `weight_norm` API in root models | Warning noise | Fixed |

### Unused / legacy code

- `train_paper.py`, `model_paper.py`, and `preprocess_paper.py` should be treated as legacy lightweight demonstrations.
- The canonical paper reproduction path is `paper_exp/`.
- `benchmark.py` is useful for quick profiling, but it includes reference values rather than implementing all external baselines.

---

## 3. Research validation against the referenced paper

### Implemented in spirit

- 1D CNN front-end.
- TCN blocks with dilated convolutions.
- LSTM temporal modelling.
- Additive attention.
- SOH regression with MSE loss in the paper experiment.
- ICA (`dQ/dV`), DV (`dV/dQ`), and DC (`dI/dV`) style features.
- Paper hyperparameter defaults: Adam, learning rate `1e-3`, batch size `64`, dropout `0.2`, 300 epochs, early stopping.

### Deviations

| Paper claim / method | Repository status |
| --- | --- |
| NASA + Oxford + CALCE paper validation | Parsers and workflow exist, but raw data and verified outputs are not committed |
| Reported SOH RMSE `0.021`, R2 `0.983` | Targets are recorded, but not reproduced in committed artefacts |
| 0.35M parameters | `paper_exp/model.py` is close at about `0.356M` |
| Embedded latency/energy | Estimated locally; hardware-dependent and not directly equivalent |
| Full baseline comparison against Transformer/XGBoost/CNN/LSTM | Not implemented as trainable baselines |
| Interpretability with attention/Grad-CAM | Attention export exists; Grad-CAM-style analysis is not implemented |

### Reproduction classification

**Partial reproduction.** The architecture and preprocessing are aligned with the paper at a high level, but the repository does not yet prove empirical reproduction of paper metrics.

---

## 4. Experimental review

### Train/test split

- `paper_exp/train.py` uses fold splits by dataset segments.
- This is better than the legacy root scripts, but it is not a strict leave-cell-out or leave-vehicle-out protocol.
- For a dissertation, a stronger protocol should group by `cell_id` or `vehicle_id` to test generalisation.

### Evaluation metrics

Implemented:

- RMSE
- MAE
- R2
- parameter count
- latency estimate
- energy estimate
- RUL RMSE for the modified experiment

Missing:

- confidence intervals over repeated seeds;
- per-chemistry results;
- per-cell / per-vehicle breakdown;
- error distribution plots;
- calibration plots;
- monotonicity violation statistics;
- statistical significance testing against baselines.

### Data leakage risks

Potential leakage remains if random or segment folds include samples from the same cell/vehicle in both train and validation folds. For battery degradation, this can inflate performance because adjacent cycles are highly correlated.

Recommended fix:

- implement `GroupKFold`-style splitting by `cell_id`, `pack_id`, or `vehicle_id`;
- report both in-domain cycle split and out-of-cell/out-of-vehicle split.

### Overfitting risks

- 300 epochs on correlated cycle data may overfit without group validation.
- Synthetic fallback results should not be used as evidence of real diagnostic performance.
- Hyperparameters have not been justified through validation or ablation.

---

## 5. Dissertation alignment

### Current level

The repository supports an MSc implementation chapter but is not yet sufficient as a complete distinction-level dissertation artefact.

### Missing dissertation evidence

Figures:

- full preprocessing pipeline diagram;
- model architecture figure with tensor dimensions;
- SOH prediction vs true SOH plots;
- residual histograms;
- attention heatmaps;
- learning curves;
- per-dataset/per-cell performance plots.

Tables:

- dataset summary table;
- architecture parameter table;
- hyperparameter table;
- paper vs reproduction comparison table;
- ablation table: with/without TCN, LSTM, attention, monotonicity, RUL head;
- baseline comparison table: CNN, LSTM, CNN-LSTM, XGBoost, Transformer if claimed.

Analyses:

- data provenance and ethics;
- limitations of Kaggle vs NASA/Oxford/CALCE;
- discussion of why paper metrics may not reproduce;
- hardware-normalised efficiency discussion;
- threat-to-validity section.

---

## 6. README review

The README has been rewritten to:

- remove unsupported "full state-of-the-art reproduction" language;
- identify `paper_exp/` as the canonical paper workflow;
- distinguish paper reproduction, legacy demos, and modified MSc contribution;
- document installation and smoke checks;
- document Kaggle dataset preparation;
- add paper-first comparison commands;
- add architecture diagrams;
- state reproducibility limitations honestly.

This is now much closer to publication-quality repository documentation.

---

## 7. Supervisor assessment grade

### Pass / Merit / Distinction decision

**Current repository artefact grade: Merit**

Rationale:

- It exceeds a basic pass because it contains a working deep learning implementation, a real dataset conversion path, and a structured comparison workflow.
- It is not yet distinction-level because it lacks committed experimental evidence, formal testing, robust group validation, ablations, and a complete dissertation artefact.

### What would make it Distinction-level

1. Commit reproducible full-run metrics and logs.
2. Add tests for preprocessing, dataset conversion, model shape, metrics, and comparison output.
3. Add group-wise validation by cell/vehicle to reduce leakage risk.
4. Add ablation studies and baseline models.
5. Add trained checkpoints or model cards.
6. Add figures and tables for the dissertation.
7. Add a clear limitations section explaining why paper metrics may not reproduce.
8. Add a lockfile or Docker image.

---

## 8. Priority improvement list

### P0 - Academic integrity and reproducibility

1. Do not claim exact paper reproduction unless full metrics are reproduced on real data.
2. Commit a final `comparison.md` from the full run or include it in dissertation appendices.
3. Add data provenance and licensing notes for the Kaggle and public battery datasets.

### P1 - Experimental validity

1. Implement group-wise train/validation/test splits by cell/vehicle.
2. Add repeated-seed experiments.
3. Add baseline implementations or remove unsupported baseline claims.
4. Save checkpoints and prediction CSVs.

### P2 - Code quality

1. Add a `tests/` suite.
2. Add `pyproject.toml` with linting/formatting.
3. Add CI that runs smoke tests.
4. Reduce duplication between root legacy scripts and `paper_exp/`.

### P3 - Dissertation presentation

1. Add architecture and preprocessing figures.
2. Add attention visualisation examples.
3. Add tables for dataset statistics, hyperparameters, and ablations.
4. Add supervisor-style limitations and threat-to-validity section.

---

## 9. Exact code fixes already applied in this review cycle

1. `paper_exp/train.py`
   - uses the actual loaded tensor sequence length for benchmarking;
   - warns when requested `seq_len` differs from processed feature length.

2. `paper_exp/run_comparison.py`
   - smoke mode now auto-generates demo Kaggle-style data if no processed Kaggle file is present.

3. `README.md`
   - rewritten to publication-quality, honest, reproducible documentation.

4. `docs/AUDIT_REPORT.md`
   - this formal audit report added for MSc supervisor review.

---

## Final recommendation

Use this repository as the implementation appendix for the MSc dissertation, but do not submit it as a final distinction-level evidence package until the full experiment outputs, group validation, tests, and dissertation analyses are added.

