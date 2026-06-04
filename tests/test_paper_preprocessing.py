"""Unit tests for paper-exact feature pipeline (Experiment A)."""

import numpy as np

from experiments.paper_config import PAPER_SEQ_LEN
from experiments.paper_preprocessing import extract_paper_cycle_tensor, sanitize_feature_tensor


def test_extract_paper_cycle_tensor_shape():
    n = 80
    v = np.linspace(2.5, 4.2, n)
    i = np.ones(n) * 1.5
    q = np.linspace(0, 2.0, n)
    tensor = extract_paper_cycle_tensor(v, i, q, seq_len=PAPER_SEQ_LEN)
    assert tensor is not None
    assert tensor.shape == (3, PAPER_SEQ_LEN)
    assert np.all(np.isfinite(tensor))


def test_sanitize_feature_tensor_replaces_nan():
    bad = np.array([1.0, np.nan, np.inf], dtype=np.float32)
    clean = sanitize_feature_tensor(bad)
    assert np.all(np.isfinite(clean))


def test_ica_dv_dc_channels_differ():
    n = 100
    v = np.linspace(2.5, 4.2, n)
    i = 1.5 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n))
    q = np.linspace(0, 2.0, n) ** 1.1
    tensor = extract_paper_cycle_tensor(v, i, q, seq_len=PAPER_SEQ_LEN)
    assert not np.allclose(tensor[0], tensor[1])
    assert not np.allclose(tensor[0], tensor[2])
