import json
import os

import torch

from experiments.config import CHECKPOINT_DIR, RESULTS_DIR


def ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_checkpoint(model, path, metadata=None):
    ensure_dirs()
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(model, path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return payload.get("metadata", {})


def save_json(data, filename):
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path
