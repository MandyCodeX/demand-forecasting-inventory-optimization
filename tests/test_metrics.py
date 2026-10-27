"""Tests for WAPE, RMSSE, MAE, RMSE, and Interval Coverage."""

import pytest
import numpy as np
from src.metrics import wape, rmsse, mae, rmse, interval_coverage, evaluate_forecast


def test_wape_known_example():
    # True demand: [10, 20, 30, 40], Total = 100
    # Predicted:   [12, 18, 33, 38], Errors = [2, 2, 3, 2], Total error = 9
    # WAPE = 9 / 100 = 0.09
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])
    w = wape(y_true, y_pred)
    assert np.isclose(w, 0.09)


def test_rmsse_known_example():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])
    y_hist = np.array([8.0, 9.0, 10.0, 11.0, 10.0])

    r = rmsse(y_true, y_pred, y_hist)
    assert r > 0.0
    assert np.isfinite(r)


def test_interval_coverage():
    y_true = np.array([10.0, 15.0, 25.0, 30.0])
    y_lower = np.array([8.0, 12.0, 26.0, 28.0])
    y_upper = np.array([12.0, 18.0, 30.0, 35.0])
    # Item 1: 10 in [8, 12] (yes)
    # Item 2: 15 in [12, 18] (yes)
    # Item 3: 25 not in [26, 30] (no)
    # Item 4: 30 in [28, 35] (yes)
    # Expected coverage = 3/4 = 0.75
    cov = interval_coverage(y_true, y_lower, y_upper)
    assert np.isclose(cov, 0.75)


def test_evaluate_forecast():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    res = evaluate_forecast(y_true, y_pred)
    assert res["WAPE"] == 0.0
    assert res["MAE"] == 0.0
    assert res["RMSE"] == 0.0
