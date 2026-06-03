import os

import numpy as np
import pytest


def _has_nasa():
    d = os.path.join(os.getcwd(), "data", "NASA")
    return os.path.isdir(d) and any(f.endswith(".mat") for f in os.listdir(d))


def _has_oxford():
    p = os.path.join(os.getcwd(), "data", "Oxford", "Oxford_Battery_Degradation_Dataset_1.mat")
    return os.path.isfile(p) and os.path.getsize(p) > 1_000_000


def _has_calce():
    from experiments.calce_loader import _list_cell_dirs

    return len(_list_cell_dirs(os.path.join("data", "CALCE"))) > 0


@pytest.mark.skipif(not _has_nasa(), reason="NASA .mat files not downloaded")
def test_nasa_loader_returns_valid_soh_range():
    from experiments.nasa_loader import iter_nasa_discharge_cycles

    cycles = list(iter_nasa_discharge_cycles(os.path.join("data", "NASA")))
    assert len(cycles) >= 100
    soh = [c[3] for c in cycles]
    assert min(soh) >= 0.5
    assert max(soh) <= 1.05


@pytest.mark.skipif(not _has_oxford(), reason="Oxford .mat not downloaded")
def test_oxford_loader_returns_valid_cycles():
    from experiments.oxford_loader import iter_oxford_characterisation_cycles

    mat_path = os.path.join("data", "Oxford", "Oxford_Battery_Degradation_Dataset_1.mat")
    cycles = list(iter_oxford_characterisation_cycles(mat_path))
    assert len(cycles) >= 100
    soh = [c[3] for c in cycles]
    assert min(soh) >= 0.5
    assert max(soh) <= 1.05


@pytest.mark.skipif(not _has_calce(), reason="CALCE xlsx logs not downloaded")
def test_calce_loader_returns_valid_cycles():
    from experiments.calce_loader import iter_calce_all_cells

    cycles = list(iter_calce_all_cells(os.path.join("data", "CALCE")))
    assert len(cycles) >= 100
    soh = np.array([c[3] for c in cycles])
    assert np.isfinite(soh).all()
    assert soh.max() <= 1.05


def test_provenance_detects_missing_data():
    from experiments.provenance import detect_data_sources

    sources = detect_data_sources()
    assert set(sources.keys()) == {"NASA", "Oxford", "CALCE"}
    for dataset, info in sources.items():
        assert "experiment_a_paper" in info
        assert "experiment_b_msc" in info
