import json
import os
import time

import numpy as np
import torch

from experiments.config import PAPER_REFERENCE, RESULTS_DIR, SEQ_LEN
from experiments.io_utils import ensure_dirs, save_json
from model import BatteryHealthPredictor
from model_paper import BatterySOHPredictorPaper


def estimate_model_macs(model_type="advanced", seq_len=SEQ_LEN):
    macs = 0
    l_conv = seq_len
    macs += 3 * 32 * 5 * l_conv
    l_pool = seq_len // 2
    macs += (32 * 32 * 3 * l_pool) * 2
    macs += (32 * 64 * 3 * l_pool) + (64 * 64 * 3 * l_pool)
    macs += 32 * 64 * 1 * l_pool
    lstm_macs_per_step = 4 * (64 * 64 + 64 * 64)
    macs += lstm_macs_per_step * l_pool
    macs += (64 * 32) * l_pool
    macs += (32 * 1) * l_pool
    macs += (64 * 32) + (32 * 1)
    if model_type != "paper":
        macs += (64 * 32) + (32 * 1)
    return macs


def benchmark_model(model, device, num_runs=300):
    model.eval()
    dummy_input = torch.randn(1, 3, 100).to(device)

    for _ in range(30):
        with torch.no_grad():
            _ = model(dummy_input)

    latencies = []
    if device.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(num_runs):
            starter.record()
            with torch.no_grad():
                _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))
    else:
        for _ in range(num_runs):
            t_start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

    return float(np.mean(latencies))


def run_benchmark():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paper = BatterySOHPredictorPaper().to(device)
    model_msc = BatteryHealthPredictor().to(device)

    params_paper = sum(p.numel() for p in model_paper.parameters() if p.requires_grad)
    params_msc = sum(p.numel() for p in model_msc.parameters() if p.requires_grad)

    latency_paper = benchmark_model(model_paper, device)
    latency_msc = benchmark_model(model_msc, device)
    power_bms_w = 0.103

    stats = {
        "device": device.type,
        "paper_reproduction": {
            "params_m": params_paper / 1e6,
            "latency_ms": latency_paper,
            "energy_mj": power_bms_w * latency_paper,
            "macs": estimate_model_macs("paper"),
        },
        "msc_proposed": {
            "params_m": params_msc / 1e6,
            "latency_ms": latency_msc,
            "energy_mj": power_bms_w * latency_msc,
            "macs": estimate_model_macs("advanced"),
        },
        "published_baseline": PAPER_REFERENCE,
    }

    save_json(stats, "computational_benchmark.json")

    ref_t = PAPER_REFERENCE["transformer"]
    ref_p = PAPER_REFERENCE["paper_hybrid"]

    print("\n" + "=" * 88)
    print(f"{'COMPUTATIONAL BENCHMARK':^88}")
    print("=" * 88)
    print(f" {'Metric':<22} | {'Transformer (pub.)':<18} | {'Paper repro. (ours)':<20} | {'MSc PI-MT (ours)':<16}")
    print("-" * 88)
    print(f" {'Parameters (M)':<22} | {ref_t['params_m']:<18.2f} | {stats['paper_reproduction']['params_m']:<20.4f} | {stats['msc_proposed']['params_m']:<16.4f}")
    print(f" {'Latency (ms)':<22} | {ref_t['latency_ms']:<18.1f} | {stats['paper_reproduction']['latency_ms']:<20.3f} | {stats['msc_proposed']['latency_ms']:<16.3f}")
    print(f" {'Energy (mJ)':<22} | {ref_t['energy_mj']:<18.2f} | {stats['paper_reproduction']['energy_mj']:<20.3f} | {stats['msc_proposed']['energy_mj']:<16.3f}")
    print(f" {'Published SOH RMSE':<22} | {ref_t['soh_rmse']:<18.3f} | {ref_p['soh_rmse']:<20.3f} | {'SOH + RUL':<16}")
    print("=" * 88)

    exp_report = os.path.join(RESULTS_DIR, "experiment_comparison_report.json")
    if os.path.exists(exp_report):
        with open(exp_report, encoding="utf-8") as handle:
            exp = json.load(handle)
        print("\nLinked experiment SOH RMSE (from latest run_experiments.py):")
        for row in exp.get("experiment_a_paper_reproduction", []):
            print(f"  Paper repro. | {row['dataset']}: {row['metrics']['rmse']:.4f}")
        for row in exp.get("experiment_b_msc_extension", []):
            soh = row["metrics"]["soh"]["rmse"]
            rul = row["metrics"]["rul"]["rmse"]
            print(f"  MSc PI-MT    | {row['dataset']}: SOH={soh:.4f}, RUL={rul:.2f}")

    print(f"\nSaved to: {os.path.join(RESULTS_DIR, 'computational_benchmark.json')}\n")
    return stats


if __name__ == "__main__":
    run_benchmark()
