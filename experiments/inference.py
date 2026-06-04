"""Load paper checkpoints and produce predictions aligned with the evaluation report.

Reported metrics in ``paper_experiment_report.json`` use stratified 5-fold CV (pooled OOF).
Figures use the same OOF predictions when fold checkpoints exist.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.config import CHECKPOINT_DIR, RESULTS_DIR
from experiments.cv import chronological_split, stratified_kfold_splits
from experiments.io_utils import load_checkpoint
from experiments.paper_config import PAPER_SEQ_LEN
from experiments.runtime import get_device
from experiments.trainer import PaperDataset, split_indices, _evaluate_paper_model
from model_paper import build_paper_model
from preprocess_paper import PaperDatasetLoader


def _checkpoint_for_fold(dataset_name: str, fold: int, report_entry: Optional[dict] = None) -> str:
    base = os.path.join(CHECKPOINT_DIR, f"paper_{dataset_name.lower()}_fold{fold}.pt")
    if os.path.isfile(base):
        return base
    if report_entry and report_entry.get("checkpoint"):
        path = report_entry["checkpoint"].replace("/", os.sep)
        if os.path.isfile(path):
            return path
    return base


def predict_cv_oof(
    dataset_name: str,
    report_path: Optional[str] = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Pooled out-of-fold predictions (matches stratified 5-fold CV metrics)."""
    device = device or get_device()
    features, soh = PaperDatasetLoader.load_dataset(dataset_name, seq_len=PAPER_SEQ_LEN)
    features = np.asarray(features, dtype=np.float32)
    soh = np.asarray(soh, dtype=np.float32)

    report_entry = None
    n_folds = 5
    if report_path and os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)
        for r in report.get("results", []):
            if r["dataset"] == dataset_name:
                report_entry = r
                n_folds = len(r.get("fold_results", [])) or 5
                break

    n_samples = len(soh)
    y_true_all = np.full(n_samples, np.nan, dtype=np.float64)
    y_pred_all = np.full(n_samples, np.nan, dtype=np.float64)
    folds_used = 0

    model = build_paper_model(seq_len=PAPER_SEQ_LEN)
    for fold_i, (_, val_idx) in enumerate(stratified_kfold_splits(soh), start=1):
        if fold_i > n_folds:
            break
        ckpt = _checkpoint_for_fold(dataset_name, fold_i, report_entry)
        if not os.path.isfile(ckpt):
            continue
        load_checkpoint(model, ckpt, device)
        val_ds = PaperDataset(features[val_idx], soh[val_idx])
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        _, y_true, y_pred = _evaluate_paper_model(model, val_loader, device)
        y_true_all[val_idx] = y_true
        y_pred_all[val_idx] = y_pred
        folds_used += 1

    if folds_used == 0:
        return predict_chronological_holdout(dataset_name, device=device)

    valid = np.isfinite(y_pred_all)
    order = np.argsort(np.where(valid, np.arange(n_samples), n_samples + 1))

    return {
        "dataset": dataset_name,
        "eval_note": "stratified_5fold_cv_pooled_oof",
        "cycles": np.arange(n_samples)[order].tolist(),
        "soh_true": y_true_all[order].tolist(),
        "soh_pred": y_pred_all[order].tolist(),
        "folds_used": folds_used,
    }


def predict_chronological_holdout(
    dataset_name: str,
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Chronological 80/20 hold-out (supplementary; used when CV checkpoints missing)."""
    device = device or get_device()
    features, soh = PaperDatasetLoader.load_dataset(dataset_name, seq_len=PAPER_SEQ_LEN)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"paper_{dataset_name.lower()}.pt")

    split_idx = split_indices(len(features))
    val_ds = PaperDataset(features[split_idx:], soh[split_idx:])
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = build_paper_model(seq_len=PAPER_SEQ_LEN).to(device)
    if os.path.isfile(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device)
    model.eval()

    _, y_true, y_pred = _evaluate_paper_model(model, loader, device)
    cycles = np.arange(split_idx, split_idx + len(y_true))

    return {
        "dataset": dataset_name,
        "split_idx": int(split_idx),
        "eval_note": "chronological_80_20_holdout",
        "cycles": cycles.tolist(),
        "soh_true": y_true.tolist(),
        "soh_pred": y_pred.tolist(),
        "checkpoint": checkpoint_path,
    }


def collect_all_predictions(
    datasets: Iterable[str] = ("NASA", "Oxford", "CALCE"),
    report_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect OOF predictions per dataset (falls back to chronological if no checkpoints)."""
    if report_path is None:
        report_path = os.path.join(RESULTS_DIR, "paper_experiment_report.json")
    return {name: predict_cv_oof(name, report_path=report_path) for name in datasets}
