"""Computational benchmark for paper reproduction model."""

import os
import time

import numpy as np
import torch

from experiments.config import PAPER_REFERENCE, RESULTS_DIR
from experiments.io_utils import ensure_dirs, save_json
from experiments.paper_config import PAPER_SEQ_LEN
from experiments.runtime import configure_runtime
from model_paper import build_paper_model

EDGE_POWER_WATTS = 0.103


def benchmark_model(model, device, num_runs=300, seq_len=PAPER_SEQ_LEN):
    model.eval()
    dummy_input = torch.randn(1, 3, seq_len).to(device)
    for _ in range(30):
        with torch.no_grad():
            _ = model(dummy_input)
    latencies = []
    for _ in range(num_runs):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies.append((time.perf_counter() - t_start) * 1000.0)
    return float(np.mean(latencies))


def run_benchmark(force_cpu=False):
    ensure_dirs()
    device = configure_runtime(force_cpu=force_cpu)
    model = build_paper_model(seq_len=PAPER_SEQ_LEN).to(device)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    latency_ms = benchmark_model(model, device)
    stats = {
        "device": device.type,
        "paper_reproduction": {
            "params_m": params_m,
            "latency_ms": latency_ms,
            "energy_mj": EDGE_POWER_WATTS * latency_ms,
        },
        "published_baseline": PAPER_REFERENCE,
    }
    save_json(stats, "computational_benchmark.json")
    print(f"Paper model: {params_m:.4f} M params, {latency_ms:.3f} ms latency")
    return stats


if __name__ == "__main__":
    run_benchmark()
