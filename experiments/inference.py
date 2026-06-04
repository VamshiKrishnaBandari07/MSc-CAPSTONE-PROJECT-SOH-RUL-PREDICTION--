"""Load paper checkpoints and produce validation predictions for figures.

Note: Figures use a single chronological 80/20 hold-out for visualization.
Reported RMSE in ``paper_experiment_report.json`` uses stratified 5-fold CV.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.io_utils import load_checkpoint
from experiments.paper_config import PAPER_SEQ_LEN
from experiments.runtime import get_device
from experiments.trainer import PaperDataset, split_indices
from model_paper import build_paper_model
from preprocess_paper import PaperDatasetLoader


def predict_paper_validation(
    dataset_name: str,
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Run inference on the chronological validation split (figure generation)."""
    device = device or get_device()
    features, soh = PaperDatasetLoader.load_dataset(dataset_name, seq_len=PAPER_SEQ_LEN)

    if checkpoint_path is None:
        checkpoint_path = os.path.join("checkpoints", f"paper_{dataset_name.lower()}.pt")

    split_idx = split_indices(len(features))
    val_ds = PaperDataset(features[split_idx:], soh[split_idx:])
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = build_paper_model(seq_len=PAPER_SEQ_LEN).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            pred, _ = model(batch_x)
            preds.append(pred.cpu().numpy())
            targets.append(batch_y.numpy())

    y_pred = np.concatenate(preds, axis=0).flatten()
    y_true = np.concatenate(targets, axis=0).flatten()
    cycles = np.arange(split_idx, split_idx + len(y_true))

    return {
        "dataset": dataset_name,
        "split_idx": int(split_idx),
        "eval_note": "chronological_80_20_holdout_for_figures_only",
        "cycles": cycles.tolist(),
        "soh_true": y_true.tolist(),
        "soh_pred": y_pred.tolist(),
        "checkpoint": checkpoint_path,
    }


def collect_all_predictions(
    datasets: Iterable[str] = ("NASA", "Oxford", "CALCE"),
) -> Dict[str, Dict[str, Any]]:
    """Collect validation predictions for all datasets."""
    return {name: predict_paper_validation(name) for name in datasets}
