"""Generate side-by-side comparison reports for capstone experiments."""

from experiments.config import PAPER_REFERENCE


def _pct_delta(ours, reference):
    if reference == 0:
        return 0.0
    return ((ours - reference) / reference) * 100.0


def print_comparison_report(paper_results, msc_results, benchmark_stats, ablation_results=None):
    print("\n" + "=" * 96)
    print(f"{'MSc CAPSTONE - EXPERIMENT COMPARISON REPORT':^96}")
    print("=" * 96)

    print("\n[Experiment A] Paper-exact reproduction vs published baselines")
    print("-" * 96)
    print(
        f"{'Dataset':<10} | {'Our SOH RMSE':<14} | {'Paper Hybrid RMSE':<18} | "
        f"{'Transformer RMSE':<16} | {'vs Paper (%)':<14} | {'Mono Viol.':<10}"
    )
    print("-" * 96)

    paper_ref_rmse = PAPER_REFERENCE["paper_hybrid"]["soh_rmse"]
    transformer_rmse = PAPER_REFERENCE["transformer"]["soh_rmse"]

    for result in paper_results:
        ds = result["dataset"]
        rmse = result["metrics"]["rmse"]
        mono = result["metrics"]["mono_violation_rate"]
        delta = _pct_delta(rmse, paper_ref_rmse)
        print(
            f"{ds:<10} | {rmse:<14.4f} | {paper_ref_rmse:<18.3f} | "
            f"{transformer_rmse:<16.3f} | {delta:>+13.1f}% | {mono:<10.2%}"
        )

    print("\n[Experiment B] MSc extension (joint SOH + RUL + physics-informed loss)")
    print("-" * 96)
    print(
        f"{'Dataset':<10} | {'SOH RMSE':<10} | {'SOH R2':<8} | {'RUL RMSE':<12} | "
        f"{'Mono Viol.':<10} | {'Paper SOH RMSE':<14} | {'SOH Gain vs Paper':<16}"
    )
    print("-" * 96)

    paper_by_ds = {r["dataset"]: r["metrics"]["rmse"] for r in paper_results}
    for result in msc_results:
        ds = result["dataset"]
        soh = result["metrics"]["soh"]
        rul = result["metrics"]["rul"]
        paper_soh = paper_by_ds.get(ds, paper_ref_rmse)
        gain = paper_soh - soh["rmse"]
        print(
            f"{ds:<10} | {soh['rmse']:<10.4f} | {soh['r2']:<8.4f} | {rul['rmse']:<12.2f} | "
            f"{soh['mono_violation_rate']:<10.2%} | {paper_soh:<14.4f} | {gain:>+15.4f}"
        )

    if ablation_results:
        print("\n[Experiment C] Ablation - MSc without physics monotonicity penalty")
        print("-" * 96)
        print(f"{'Dataset':<10} | {'SOH RMSE (no physics)':<22} | {'SOH RMSE (with physics)':<24} | {'Mono reduction':<14}")
        print("-" * 96)
        msc_by_ds = {r["dataset"]: r for r in msc_results}
        for ablation in ablation_results:
            ds = ablation["dataset"]
            full = msc_by_ds[ds]
            mono_drop = ablation["metrics"]["soh"]["mono_violation_rate"] - full["metrics"]["soh"]["mono_violation_rate"]
            print(
                f"{ds:<10} | {ablation['metrics']['soh']['rmse']:<22.4f} | "
                f"{full['metrics']['soh']['rmse']:<24.4f} | {mono_drop:>+13.2%}"
            )

    print("\n[Computational profile]")
    print("-" * 96)
    print(
        f"{'Model':<28} | {'Params (M)':<12} | {'Latency (ms)':<14} | {'Energy (mJ)':<12} | {'Targets':<12}"
    )
    print("-" * 96)
    print(
        f"{'Transformer (published)':<28} | "
        f"{PAPER_REFERENCE['transformer']['params_m']:<12.2f} | "
        f"{PAPER_REFERENCE['transformer']['latency_ms']:<14.1f} | "
        f"{PAPER_REFERENCE['transformer']['energy_mj']:<12.2f} | SOH"
    )
    print(
        f"{'Paper reproduction (ours)':<28} | "
        f"{benchmark_stats['paper']['params_m']:<12.4f} | "
        f"{benchmark_stats['paper']['latency_ms']:<14.3f} | "
        f"{benchmark_stats['paper']['energy_mj']:<12.3f} | SOH"
    )
    print(
        f"{'MSc proposed (PI-MT)':<28} | "
        f"{benchmark_stats['msc']['params_m']:<12.4f} | "
        f"{benchmark_stats['msc']['latency_ms']:<14.3f} | "
        f"{benchmark_stats['msc']['energy_mj']:<12.3f} | SOH+RUL"
    )
    print("=" * 96 + "\n")


def build_summary_payload(paper_results, msc_results, benchmark_stats, ablation_results=None):
    return {
        "paper_reference": PAPER_REFERENCE,
        "experiment_a_paper_reproduction": paper_results,
        "experiment_b_msc_extension": msc_results,
        "experiment_c_ablation_no_physics": ablation_results or [],
        "computational_profile": benchmark_stats,
    }
