"""Device selection and CPU-friendly runtime tuning."""

import os

import torch

_force_cpu = False


def set_force_cpu(enabled=True):
    """Force all training to use CPU (e.g. when no GPU or for reproducibility)."""
    global _force_cpu
    _force_cpu = bool(enabled)


def get_device():
    """Return cuda device if available, unless CPU is forced."""
    if _force_cpu or os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_runtime(force_cpu=False):
    """
    Configure PyTorch for the active device and print runtime info.
    Call once at the start of each experiment script.
    """
    if force_cpu:
        set_force_cpu(True)

    device = get_device()

    if device.type == "cpu":
        threads = min(os.cpu_count() or 4, 8)
        torch.set_num_threads(threads)
        # Avoid overhead when CUDA build is installed but unused
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.enabled = False
        print(f"[Runtime] CPU mode — PyTorch threads: {threads} (use --cpu or FORCE_CPU=1 to force)")
    else:
        name = torch.cuda.get_device_name(0)
        print(f"[Runtime] GPU mode — {name}")

    return device


def paper_batch_size(device=None):
    from experiments.paper_config import PAPER_BATCH_SIZE, PAPER_BATCH_SIZE_CPU

    device = device or get_device()
    return PAPER_BATCH_SIZE_CPU if device.type == "cpu" else PAPER_BATCH_SIZE


def msc_batch_size(device=None):
    from experiments.config import BATCH_SIZE, BATCH_SIZE_CPU

    device = device or get_device()
    return BATCH_SIZE_CPU if device.type == "cpu" else BATCH_SIZE
