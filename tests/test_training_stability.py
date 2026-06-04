import numpy as np
import torch

from experiments.training_stability import augment_features, apply_fold_scaler, fit_fold_scaler, sanitize_predictions


def test_sanitize_predictions_clips_nan():
    y = sanitize_predictions(np.array([np.nan, 1.5, -0.2]))
    assert np.all(np.isfinite(y))
    assert y[0] == 0.5
    assert y[1] == 1.0
    assert y[2] == 0.0


def test_fold_scaler_uses_train_only():
    features = np.random.rand(20, 3, 10).astype(np.float32)
    train_idx = np.arange(15)
    val_idx = np.arange(15, 20)
    mins, maxs = fit_fold_scaler(features, train_idx)
    scaled = apply_fold_scaler(features, mins, maxs)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    assert scaled[val_idx].max() <= 1.0


def test_augment_features_stays_in_unit_interval():
    x = torch.ones(4, 3, 10) * 0.5
    out = augment_features(x, voltage_jitter_scale=0.01, feature_noise=0.01)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
