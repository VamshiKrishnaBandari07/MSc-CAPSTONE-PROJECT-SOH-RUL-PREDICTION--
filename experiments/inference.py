"""Load saved checkpoints and produce validation-set predictions for plotting."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.config import CHECKPOINT_DIR, NUM_CYCLES, SEQ_LEN
from experiments.io_utils import load_checkpoint
from experiments.trainer import MScDataset, PaperDataset, split_indices
from model import BatteryHealthPredictor
from model_paper import BatterySOHPredictorPaper
from preprocess import BatteryDatasetLoader
from preprocess_paper import PaperDatasetLoader


def _predict_paper(features, soh, checkpoint_path, device):
    split_idx = split_indices(len(features))
    val_ds = PaperDataset(features[split_idx:], soh[split_idx:])
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = BatterySOHPredictorPaper(input_features=3).to(device)
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
        "split_idx": split_idx,
        "cycles": cycles.tolist(),
        "soh_true": y_true.tolist(),
        "soh_pred": y_pred.tolist(),
    }


def _predict_msc(model_cls, features, soh, rul, checkpoint_path, device):
    split_idx = split_indices(len(features))
    max_rul = float(np.max(rul)) if len(rul) else 1.0
    val_ds = MScDataset(features[split_idx:], soh[split_idx:], rul[split_idx:], max_rul)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = model_cls(input_features=3).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    soh_preds, soh_true = [], []
    rul_preds, rul_true = [], []
    with torch.no_grad():
        for batch_x, batch_soh, batch_rul in loader:
            batch_x = batch_x.to(device)
            pred_soh, pred_rul, _ = model(batch_x)
            soh_preds.append(pred_soh.cpu().numpy())
            soh_true.append(batch_soh.numpy())
            rul_preds.append((pred_rul * max_rul).cpu().numpy())
            rul_true.append((batch_rul * max_rul).numpy())

    y_soh_pred = np.concatenate(soh_preds, axis=0).flatten()
    y_soh_true = np.concatenate(soh_true, axis=0).flatten()
    y_rul_pred = np.concatenate(rul_preds, axis=0).flatten()
    y_rul_true = np.concatenate(rul_true, axis=0).flatten()
    cycles = np.arange(split_idx, split_idx + len(y_soh_true))

    return {
        "split_idx": split_idx,
        "max_rul": max_rul,
        "cycles": cycles.tolist(),
        "soh_true": y_soh_true.tolist(),
        "soh_pred": y_soh_pred.tolist(),
        "rul_true": y_rul_true.tolist(),
        "rul_pred": y_rul_pred.tolist(),
    }


def collect_dataset_predictions(dataset_name, paper_ckpt=None, msc_ckpt=None, ablation_ckpt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paper_features, paper_soh = PaperDatasetLoader.load_dataset(
        dataset_name, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN
    )
    msc_features, msc_soh, msc_rul = BatteryDatasetLoader.load_dataset(
        dataset_name, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN
    )

    if paper_ckpt is None:
        paper_ckpt = os.path.join(CHECKPOINT_DIR, f"paper_{dataset_name.lower()}.pt")
    if msc_ckpt is None:
        msc_ckpt = os.path.join(CHECKPOINT_DIR, f"msc_{dataset_name.lower()}.pt")
    if ablation_ckpt is None:
        ablation_ckpt = os.path.join(CHECKPOINT_DIR, f"msc_ablation_{dataset_name.lower()}.pt")

    paper_preds = _predict_paper(paper_features, paper_soh, paper_ckpt, device)
    msc_preds = _predict_msc(BatteryHealthPredictor, msc_features, msc_soh, msc_rul, msc_ckpt, device)

    result = {
        "dataset": dataset_name,
        "paper": paper_preds,
        "msc": msc_preds,
    }

    if os.path.exists(ablation_ckpt):
        result["msc_ablation"] = _predict_msc(
            BatteryHealthPredictor, msc_features, msc_soh, msc_rul, ablation_ckpt, device
        )

    return result


def collect_all_predictions(datasets=("NASA", "Oxford", "CALCE")):
    return {name: collect_dataset_predictions(name) for name in datasets}
