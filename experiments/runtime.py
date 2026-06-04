"""Device selection and CPU-friendly runtime tuning."""

from __future__ import annotations

import os

import torch

_force_cpu = False


def set_force_cpu(enabled: bool = True) -> None:
    """Force all training to use CPU."""
    global _force_cpu
    _force_cpu = bool(enabled)


def get_device() -> torch.device:
    """Return CUDA device if available, unless CPU is forced."""
    if _force_cpu or os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_runtime(force_cpu: bool = False) -> torch.device:
    """Configure PyTorch for the active device and print runtime info."""
    if force_cpu:
        set_force_cpu(True)

    device = get_device()

    if device.type == "cpu":
        threads = min(os.cpu_count() or 4, 8)
        torch.set_num_threads(threads)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.enabled = False
        print(f"[Runtime] CPU mode — PyTorch threads: {threads}")
    else:
        print(f"[Runtime] GPU mode — {torch.cuda.get_device_name(0)}")

    return device


def paper_batch_size(device: torch.device | None = None) -> int:
    """Batch size for paper model (smaller on CPU)."""
    from experiments.paper_config import PAPER_BATCH_SIZE, PAPER_BATCH_SIZE_CPU

    device = device or get_device()
    return PAPER_BATCH_SIZE_CPU if device.type == "cpu" else PAPER_BATCH_SIZE
